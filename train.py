"""
Special Variable:
    @WARMUP_STEPS: warmup steps for scheduler
    @TOTAL_STEPS: total steps for scheduler
"""
import random
from typing import Union

import hydra
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker, GeneralTracker
from monai.data import ThreadDataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

import wandb
from dataset import load_datalist, get_transforms
from utils import initiate, to_wandb_images, dice_score, iou_score, get_class

ENTITY = 'luisleo'


def compute_metrics(y_pred, y_true, loss, metrics, targets):
    dice = dice_score(y_pred, y_true, len(targets))
    iou = iou_score(y_pred, y_true, len(targets))

    metrics['loss'].append(loss.detach())
    metrics['dice'].append(dice.mean(dim=0))
    metrics['iou'].append(iou.mean(dim=0))


def aggregate_metrics(split, metrics, targets):
    results = {}
    for key, metric in metrics.items():
        scores = metric.aggregate()
        if key == 'loss':
            results[f"{key}/{split}"] = float(scores)
        else:
            results.update({f"{key}-{targets[idx]}/{split}": v for idx, v in enumerate(scores)})
        metric.reset()
    return results


def save_and_upload(accelerator: Accelerator, model, cfg, tag):
    if accelerator.is_main_process:
        accelerator.save_model(model, f"{cfg.save_dir}/{cfg.name}/{cfg.save_tag}-{tag}")
        if cfg.track:
            art = wandb.Artifact(f"{wandb.run.id}-{tag}", type='model', metadata={
                'task_name': cfg.name,
                'model_name': cfg.model.network.type,
                'num_of_classes': len(cfg.data.targets)
            })
            art.add_file(f"{cfg.save_dir}/{cfg.name}/{cfg.save_tag}-{tag}/pytorch_model.bin")

            tracker: Union[WandBTracker, GeneralTracker] = accelerator.get_tracker('wandb')
            tracker.tracker.log_artifact(art)
            if tag == 'best':
                tracker.tracker.link_artifact(art,
                                              f"{ENTITY}/model-registry/{cfg.model.network.type.split('.')[-1]}-{cfg.name}")


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(gradient_accumulation_steps=cfg.model.accumulation_steps,
                              log_with="wandb" if cfg.track else None, )

    if cfg.track:
        config = OmegaConf.to_container(cfg, resolve=True)
        config['equivalent_batch_size'] = config['model']['batch_size']['train'] * accelerator.num_processes * \
                                          cfg.model.accumulation_steps
        accelerator.init_trackers(cfg.name, init_kwargs={"wandb": {"config": config}})
        if accelerator.is_main_process:
            tracker: Union[WandBTracker, GeneralTracker] = accelerator.get_tracker('wandb')
            tracker.tracker.define_metric("dice*", summary="max")
            tracker.tracker.define_metric("iou*", summary="max")
            tracker.tracker.define_metric("loss*", summary="min")

    datalist = load_datalist(cfg)

    if not cfg.self_training:
        del datalist['test']

    for split, data in datalist.items():
        accelerator.print(f"{split} has {len(data)} samples")

    model = initiate(cfg.model.network, cfg=cfg, skip=True)
    if cfg.load:
        model.load_state_dict(torch.load(f"{cfg.save_dir}/{cfg.name}/{cfg.load_tag}/pytorch_model.bin"))
        accelerator.print(f"Ckeckpoint {cfg.save_dir}/{cfg.name}/{cfg.load_tag} loaded")

    dataset_class = get_class(cfg.dataset.type)
    datasets = {
        k: dataset_class(data=v if not cfg.debug else v[:5],
                         transform=get_transforms(k, cfg, accelerator.device), **cfg.dataset.params)
        for k, v in datalist.items()
    }

    dataloaders = {k: ThreadDataLoader(v, batch_size=cfg.model.batch_size[k] if k == 'train' else 1,
                                       use_thread_workers=True, buffer_size=cfg.buffer_size)
                   for k, v in datasets.items()}

    dataloaders = {k: accelerator.prepare(dataloader, device_placement=[True])
                   for k, dataloader in dataloaders.items()}

    for key, value in cfg.model.scheduler.params.items():
        if isinstance(value, str):
            if '@WARMUP_STEPS' in value:
                cfg.model.scheduler.params[key] = int(
                    float(value.replace('@WARMUP_STEPS=', '')) * len(dataloaders['train'])) * cfg.num_epochs
            elif '@TOTAL_STEPS' in value:
                cfg.model.scheduler.params[key] = len(dataloaders['train']) * cfg.num_epochs

    optim = initiate(cfg.model.optimizer, params=model.parameters())
    scheduler = initiate(cfg.model.scheduler, optimizer=optim)
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler, device_placement=[True, True, True])

    metrics = {k: CumulativeAverage() for k in ['dice', 'iou', 'loss']}
    targets = cfg.data.targets

    total_steps = len(dataloaders['train']) * cfg.num_epochs + len(dataloaders['val']) * cfg.num_epochs // cfg.val_freq
    if cfg.self_training:
        total_steps += len(dataloaders['test']) * cfg.num_epochs // cfg.refresh_freq

    best_dice_sum = 0.0
    pbar = trange(total_steps, disable=not accelerator.is_main_process)
    for epoch in range(cfg.num_epochs):

        results = {}

        ############################################################################################
        pbar.set_description("Training")

        model.train()
        for batch in dataloaders['train']:
            with accelerator.accumulate(model):
                pred, loss = model(batch)
                accelerator.backward(loss)
                optim.step()
                scheduler.step()
                optim.zero_grad()

                compute_metrics(pred.argmax(1, keepdim=True), batch['label'], loss, metrics, targets)

            pbar.update(1)

        results.update(aggregate_metrics('train', metrics, targets))

        ############################################################################################
        pbar.set_description("Validating")
        vis_batch = random.randint(0, len(dataloaders['val']) - 1)

        if epoch % cfg.val_freq == 0:
            model.eval()
            for batch_id, batch in enumerate(dataloaders['val']):
                with torch.no_grad():
                    pred = sliding_window_inference(inputs=batch['image'], roi_size=cfg.data.patch_size,
                                                    sw_batch_size=cfg.model.batch_size['val'],
                                                    predictor=model, overlap=0.7)
                    loss = model(pred, batch['label'])

                    compute_metrics(pred.argmax(1, keepdim=True), batch['label'], loss, metrics, targets)

                if batch_id == vis_batch and accelerator.is_main_process and cfg.track:
                    results.update(to_wandb_images(pred.argmax(1, keepdim=True), batch, cfg.data.targets))
                pbar.update(1)

            results.update(aggregate_metrics('val', metrics, targets))

            current_score = 0.0
            for target in cfg.data.targets[1:]:
                current_score += results[f"dice-{target}/val"]

            if current_score > best_dice_sum:
                best_dice_sum = current_score
                save_and_upload(accelerator, model, cfg, "best")

        ############################################################################################

        if cfg.track:
            for key in [x for x in results.keys() if cfg.data.targets[0] in x]:
                del results[key]
            accelerator.log(results)

        if epoch % cfg.save_freq == 0:
            save_and_upload(accelerator, model, cfg, "latest")

        if cfg.debug and epoch > 10:
            break

    accelerator.end_training()


if __name__ == '__main__':
    main()
