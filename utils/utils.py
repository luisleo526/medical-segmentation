import random
from importlib import import_module

import torch
import wandb
from monai.metrics import compute_iou, compute_generalized_dice
from omegaconf.dictconfig import DictConfig


def get_class(x):
    module = x[:x.rfind(".")]
    obj = x[x.rfind(".") + 1:]
    return getattr(import_module(module), obj)


def initiate(info, skip=False, **kwargs):
    class_type = get_class(info.type)
    class_params = {**kwargs}
    if not skip:
        for param_name in info.params:
            param = info.params[param_name]
            if type(param) is DictConfig:
                class_params.update({param_name: initiate(param)})
            else:
                class_params.update({param_name: param})
    return class_type(**class_params)


def extract_elements(lst, m):
    return lst[::len(lst) // m][:m]


def to_wandb_images(y_pred, batch, targets, slices=30):
    depth = y_pred.shape[-1]
    num_batch = y_pred.shape[0]
    target_batch = random.randint(0, num_batch - 1)
    class_labels = {idx: label for idx, label in enumerate(targets)}

    num_uniques = [batch['label'][target_batch, 0, :, :, i].unique().numel() for i in range(depth)]
    start = next((i for i in range(depth) if num_uniques[i] > 1), 0)
    end = next((depth - i for i in range(1, depth + 1) if num_uniques[depth - i] > 1), 0)
    start = max(start - 2, 0)
    end = min(end + 2, depth - 1)

    images = []
    for z in extract_elements(range(start, end + 1), slices):
        image = batch['image'][target_batch, 0, :, :, z].permute(1, 0).cpu().numpy()
        label = batch['label'][target_batch, 0, :, :, z].permute(1, 0).cpu().numpy()
        p_label = y_pred[target_batch, 0, :, :, z].permute(1, 0).cpu().numpy()

        images.append(wandb.Image(image, caption=f"image @ {z} / {depth}",
                                  masks={"ground_truth": {"mask_data": label, "class_labels": class_labels},
                                         "prediction": {"mask_data": p_label, "class_labels": class_labels}}))

    return {'images': images}


def dice_score(y_pred, y_truth, num_classes):
    return torch.nan_to_num(
        torch.cat([compute_generalized_dice(y_pred == idx, y_truth == idx) for idx in range(num_classes)], dim=-1))


def iou_score(y_pred, y_truth, num_classes):
    return torch.nan_to_num(
        torch.cat([compute_iou(y_pred == idx, y_truth == idx, False) for idx in range(num_classes)], dim=-1))


def move_bach_to_device(batch, device):
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device)
    return batch
