import random
import re
from importlib import import_module

import torch
from monai.metrics import compute_iou

import wandb


def get_class(x):
    module = x[:x.rfind(".")]
    obj = x[x.rfind(".") + 1:]
    return getattr(import_module(module), obj)


def compute_metrics(y_pred, y_true, metrics):
    for k, metric in metrics.items():
        s = re.search(r'\d', k)
        if bool(s):
            idx = int(s.group())
            result = compute_iou(y_pred == idx, y_true == idx).view(-1)
            result = torch.nan_to_num(result)
            metric.append(result)
        else:
            metric(y_pred, y_true)


def aggregate_metrics(metrics, targets, split):
    results = {}
    for k, metric in metrics.items():
        scores = metric.aggregate()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        if 'dice' in k:
            for idx, score in enumerate(scores):
                results[f"{k}-{targets[idx]}/{split}"] = score
        else:
            idx = re.search(r'\d', k).group()
            digits = len(idx) + 1
            results[f"{k[:-digits]}-{targets[int(idx)]}/{split}"] = scores
        metric.reset()
    return results


def extract_elements(lst, m):
    return lst[::len(lst) // m][:m]


def to_wandb_images(y_pred, batch, targets):
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
    for z in extract_elements(range(start, end + 1), 30):
        image = batch['image'][target_batch, 0, :, :, z].permute(1, 0).cpu().numpy()
        label = batch['label'][target_batch, 0, :, :, z].permute(1, 0).cpu().numpy()
        p_label = y_pred[target_batch, 0, :, :, z].permute(1, 0).cpu().numpy()

        images.append(wandb.Image(image, caption=f"image @ {z} / {depth}",
                                  masks={"ground_truth": {"mask_data": label, "class_labels": class_labels},
                                         "prediction": {"mask_data": p_label, "class_labels": class_labels}}))

    return {'images': images}
