"""
Special Variable:
    @WARMUP_STEPS: warmup steps for scheduler
    @TOTAL_STEPS: total steps for scheduler
"""
import random
from typing import Union

import hydra
import torch
import wandb
from accelerate import Accelerator
from accelerate.tracking import WandBTracker, GeneralTracker
from monai.data import ThreadDataLoader
from monai.data.utils import partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from dataset import load_datalist, get_transforms
from utils import initiate, to_wandb_images, dice_score, iou_score, get_class, move_bach_to_device, get_weights


def compute_metrics(y_pred, y_true, loss, metrics, targets, debug=False):
    dice = dice_score(y_pred, y_true, len(targets))
    iou = iou_score(y_pred, y_true, len(targets))

    _loss = loss.detach()
    _dice = dice.mean(dim=0)
    _iou = iou.mean(dim=0)

    metrics['loss'].append(_loss)
    metrics['dice'].append(_dice)
    metrics['iou'].append(_iou)

    if debug:
        print(_loss, _dice, _iou)


def aggregate_metrics(split, metrics, targets):
    results = {}
    for key, metric in metrics.items():
        scores = metric.aggregate()
        if key == 'loss':
            results[f"{split}:{key}"] = float(scores)
        else:
            for idx in range(len(scores)):
                results[f"{split}:{key}-{targets[idx]}"] = scores[idx]
        metric.reset()
    return results


def save_and_upload(accelerator: Accelerator, model, cfg, tag, snapshot):
    if accelerator.is_main_process:
        save_directory = f"{cfg.save_dir}/{cfg.name}/{cfg.save_tag}/{tag}"
        accelerator.save_model(model, save_directory, safe_serialization=False)
        if cfg.track:
            metadata = OmegaConf.to_container(cfg, resolve=True)
            metadata.update({'snapshot': snapshot})
            art = wandb.Artifact(f"model-{tag}", type='model', metadata=metadata)
            art.add_file(f"{save_directory}/pytorch_model.bin")

            tracker: Union[WandBTracker, GeneralTracker] = accelerator.get_tracker('wandb')
            alias = cfg.model.type.split('.')[-1]
            tracker.tracker.log_artifact(art, aliases=[alias])


@hydra.main(config_path="config", config_name="root", version_base="1.3")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(gradient_accumulation_steps=cfg.accumulation_steps,
                              log_with="wandb" if cfg.track else None, )

    if cfg.track:
        config = OmegaConf.to_container(cfg, resolve=True)
        config['equivalent_batch_size'] = config['batch_size']['train'] * accelerator.num_processes * \
                                          cfg.accumulation_steps
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

    model = None
    last_snapshot = None
    if cfg.load_from_artifact:
        api = wandb.Api()
        artifact = api.artifact(cfg.load_tag, type="model")

        if accelerator.is_local_main_process:
            artifact_dir = artifact.download()
            with open('.artifact_dir', 'w') as f:
                f.write(artifact_dir)

        accelerator.wait_for_everyone()

        with open('.artifact_dir', 'r') as f:
            artifact_dir = f.read()

        _cfg = OmegaConf.create(artifact.metadata)
        if 'snapshot' in artifact.metadata:
            last_snapshot = artifact.metadata['snapshot']
        # Replace CFG.MODEL with the artifact's model
        cfg.model = _cfg.model

        model = initiate(_cfg.model, cfg=_cfg, skip=True)
        model.load_state_dict(torch.load(artifact_dir + '/pytorch_model.bin'), strict=False)
        accelerator.print(f"Checkpoint {artifact_dir} loaded")
    else:
        model = initiate(cfg.model, cfg=cfg, skip=True)
        if cfg.load_from_local:
            model.load_state_dict(torch.load(f"{cfg.save_dir}/{cfg.name}/{cfg.load_tag}/best/pytorch_model.bin"))
            accelerator.print(f"Checkpoint {cfg.save_dir}/{cfg.name}/{cfg.load_tag}/best loaded")

    # Split datalist
    datalist = {k: partition_dataset(
        v, num_partitions=accelerator.num_processes, even_divisible=True, shuffle=True,
    )[accelerator.process_index] for k, v in datalist.items()}

    dataset_class = get_class(cfg.dataset.type)
    datasets = {
        k: dataset_class(data=v if not cfg.debug else v[:5],
                         transform=get_transforms(k, cfg), **cfg.dataset.params)
        for k, v in datalist.items()
    }

    dataloaders = {k: ThreadDataLoader(v, batch_size=cfg.batch_size[k] if k == 'train' else 1,
                                       use_thread_workers=True, buffer_size=cfg.buffer_size,
                                       num_workers=cfg.num_workers)
                   for k, v in datasets.items()}

    for key, value in cfg.scheduler.params.items():
        if isinstance(value, str):
            if '@WARMUP_STEPS' in value:
                cfg.scheduler.params[key] = int(
                    float(value.replace('@WARMUP_STEPS=', '')) * len(dataloaders['train'])) * cfg.num_epochs
            elif '@TOTAL_STEPS' in value:
                cfg.scheduler.params[key] = len(dataloaders['train']) * cfg.num_epochs

    optim = initiate(cfg.optimizer, params=model.parameters())
    scheduler = initiate(cfg.scheduler, optimizer=optim)
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler, device_placement=[True, True, True])

    metrics = {k: CumulativeAverage() for k in ['dice', 'iou', 'loss']}
    targets = cfg.data.targets

    total_steps = len(dataloaders['train']) * cfg.num_epochs + len(dataloaders['val']) * cfg.num_epochs // cfg.val_freq
    if cfg.self_training:
        total_steps += len(dataloaders['test']) * cfg.num_epochs // cfg.refresh_freq

    last_best_score = 0.0 if not last_snapshot else sum(
        [v * get_weights(k) for k, v in last_snapshot.items() if 'dice' in k])
    pbar = trange(total_steps, disable=not accelerator.is_main_process)
    for epoch in range(cfg.num_epochs):

        results = {}

        ############################################################################################
        pbar.set_description("Training")

        model.train()
        for batch in dataloaders['train']:
            batch = move_bach_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                pred, loss = model(batch)
                # accelerator.print(loss)
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
        current_score = 0.0
        result_snapshot = {}

        if epoch % cfg.val_freq == 0 and epoch > 0:
            model.eval()
            for batch_id, batch in enumerate(dataloaders['val']):
                batch = move_bach_to_device(batch, accelerator.device)
                with torch.no_grad():
                    pred = sliding_window_inference(inputs=batch['image'], roi_size=cfg.data.patch_size,
                                                    sw_batch_size=cfg.batch_size['val'],
                                                    predictor=model, overlap=cfg.eval_overlap,
                                                    mode='gaussian', padding_mode='replicate')
                    loss = model(pred, batch['label'])

                    compute_metrics(pred.argmax(1, keepdim=True), batch['label'], loss, metrics, targets)

                if batch_id == vis_batch and accelerator.is_main_process and cfg.track:
                    try:
                        _results = to_wandb_images(
                            pred.argmax(1, keepdim=True), batch, targets, cfg.slices_to_show
                        )
                        results.update(_results)
                    except Exception as e:
                        pass
                pbar.update(1)

            results.update(aggregate_metrics('val', metrics, targets))

            for target in targets:
                weight = get_weights(target)
                current_score += results[f"val:dice-{target}"] * weight
                result_snapshot[f"val:iou-{target}"] = results[f"val:iou-{target}"]
                result_snapshot[f"val:dice-{target}"] = results[f"val:dice-{target}"]

        ############################################################################################

        if cfg.track:
            # for key in [x for x in results.keys() if targets[0] in x]:
            #     del results[key]
            accelerator.log(results)

        if current_score > last_best_score:
            last_best_score = current_score
            save_and_upload(accelerator, model, cfg, "best", result_snapshot)

        if cfg.debug and epoch > 10:
            break

    accelerator.end_training()


if __name__ == '__main__':
    main()
