import random

import hydra
import torch
from accelerate import Accelerator
from monai.data import ThreadDataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, CumulativeAverage
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from dataset import load_datalist, get_transforms
from utils import get_class, to_wandb_images


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(gradient_accumulation_steps=cfg.model.accumulation_steps, log_with="wandb")

    if cfg.track:
        accelerator.init_trackers(cfg.name,
                                  init_kwargs={"wandb": {"config": OmegaConf.to_container(cfg, resolve=True)}})
        tracker = accelerator.get_tracker('wandb', unwrap=True)
        tracker.define_metric("dice*", summary="max")
        tracker.define_metric("iou*", summary="max")
        tracker.define_metric("loss", summary="min")

    datalist = load_datalist(cfg, accelerator.process_index, accelerator.num_processes)

    if not cfg.self_training:
        del datalist['test']

    datasets = {k: CacheDataset(data=v if not cfg.debug else v[:2], transform=get_transforms(k, cfg))
                for k, v in datalist.items()}

    dataloaders = {k: ThreadDataLoader(v, batch_size=cfg.model.batch_size[k] if k == 'train' else 1,
                                       num_workers=cfg.num_workers, use_thread_workers=True, pin_memory=True)
                   for k, v in datasets.items()}

    dataloaders = {k: accelerator.prepare(dataloader, device_placement=[True])
                   for k, dataloader in dataloaders.items()}

    model = get_class(cfg.model.network.type)(cfg)
    optim = get_class(cfg.model.optimizer.type)(model.parameters(), **cfg.model.optimizer.params)
    scheduler = get_class(cfg.model.scheduler.type)(optim, **cfg.model.scheduler.params)
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler, device_placement=[True, True, True])

    dice_metric = DiceMetric(reduction='mean_batch', num_classes=len(cfg.data.targets))
    loss_cumulative = CumulativeAverage()

    total_steps = len(dataloaders['train']) * cfg.num_epochs + len(dataloaders['val']) * cfg.num_epochs // cfg.val_freq
    if cfg.self_training:
        total_steps += len(dataloaders['test']) * cfg.num_epochs // cfg.test_freq

    pbar = trange(total_steps * cfg.num_epochs, disable=not accelerator.is_main_process)
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
                loss_cumulative.append(loss.detach())

            dice_metric(pred.argmax(1, keepdim=True), batch['label'])
            pbar.update(1)

        accelerator.wait_for_everyone()
        results.update(
            {f"dice-{cfg.data.targets[idx]}/train": score for idx, score in enumerate(dice_metric.aggregate())})
        results.update({'loss/train': loss_cumulative.aggregate().item()})
        dice_metric.reset()
        loss_cumulative.reset()

        ############################################################################################
        pbar.set_description("Validating")
        target_batch = random.randint(0, len(dataloaders['val']) - 1)

        model.eval()
        for batch_id, batch in enumerate(dataloaders['val'] if epoch % cfg.val_freq == 0 else []):
            with torch.no_grad():
                pred = sliding_window_inference(inputs=batch['image'], roi_size=cfg.data.patch_size,
                                                sw_batch_size=cfg.model.batch_size['val'],
                                                predictor=model, overlap=0.7)
                loss = model(pred, batch['label'])
                loss_cumulative.append(loss.detach())

            dice_metric(pred.argmax(1, keepdim=True), batch['label'])
            accelerator.print(dice_metric.get_buffer())
            if batch_id == target_batch and accelerator.is_main_process and cfg.track:
                results.update(to_wandb_images(pred.argmax(1, keepdim=True), batch, cfg.data.targets))
            pbar.update(1)

        accelerator.wait_for_everyone()
        results.update(
            {f"dice-{cfg.data.targets[idx]}/val": score for idx, score in enumerate(dice_metric.aggregate())})
        results.update({'loss/val': loss_cumulative.aggregate().item()})
        dice_metric.reset()
        loss_cumulative.reset()

        ############################################################################################

        if cfg.track:
            accelerator.log(results)

        if cfg.debug and epoch > 10:
            break

    accelerator.end_training()


if __name__ == '__main__':
    main()
