from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import torch
import wandb
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from monailabel.transform.writer import write_seg_nrrd
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import get_transforms, post_transform
from utils import initiate


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--artifacts', type=str, nargs='+', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--overlap', type=float, default=0.7)
    parser.add_argument('--merge_cls', type=str, nargs='+')
    parser.add_argument('--merge_name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    images = glob(str(args.data / '*.nii.gz'))
    args.output.mkdir(exist_ok=True, parents=True)

    merge_cls = args.merge_cls
    merge_cls = [x.lower() for x in merge_cls] if merge_cls is not None else []

    tasks = []
    for artifact_url in args.artifacts:
        api = wandb.Api()
        artifact = api.artifact(artifact_url, type="model")
        artifact_dir = artifact.download()
        cfg = OmegaConf.create(artifact.metadata)

        model = initiate(cfg.model, cfg=cfg, skip=True)
        model.load_state_dict(torch.load(artifact_dir + '/pytorch_model.bin'))
        model.cuda()
        model.eval()

        transform = get_transforms('test', cfg)

        tasks.append(dict(model=model, transform=transform, cfg=cfg))

    for image in tqdm(images):
        instance = Path(image).stem.split('.')[0]

        outputs = []
        for task_meta in tasks:
            model = task_meta['model']
            transform = task_meta['transform']
            cfg = task_meta['cfg']

            data = transform({'image': image})
            image_tensor = data['image'].unsqueeze(0).cuda()
            
            try:
                with torch.no_grad():
                    pred = sliding_window_inference(inputs=image_tensor, roi_size=cfg.data.patch_size,
                                                    sw_batch_size=args.batch_size, predictor=model, overlap=args.overlap)
                    prob = torch.nn.functional.softmax(pred, dim=1).half()

                    # Un-Batch
                    p_label = pred.argmax(1, keepdim=True).squeeze(0).cpu()

                    # Reverse the spatial transforms
                    p_label_inv = post_transform(p_label, cfg, data)
                    p_label_inv = one_hot(p_label_inv, len(cfg.data.targets), dim=0, dtype=torch.uint8)
                    p_label_inv = p_label_inv.cpu().numpy()[1:]

                    outputs.append(dict(
                        image=p_label_inv,
                        affine=data['image_meta_dict']['affine'].numpy(),
                        label=cfg.data.targets[1:]
                    ))
            except:
                continue
        
        filter_images = []
        filter_labels = []
        
        merge_data = None
        for entry in outputs:
            images = entry['image']
            labels = entry['label']
            affine = entry['affine']
            for image, label in zip(images, labels):
                if label.lower() in merge_cls:
                    if merge_data is None:
                        merge_data = image
                    else:
                        merge_data = np.logical_or(merge_data, image, dtype=np.uint8)
                else:
                    filter_images.append(image)
                    filter_labels.append(label)
                        
        if merge_data is not None:
            filter_images.append(merge_data)
            filter_labels.append(args.merge_name)
        
        filter_images = np.stack(filter_images)
        
        write_seg_nrrd(
            filter_images,
            f"{str(args.output)}/{instance}.seg.nrrd",
            np.uint8,
            outputs[0]['affine'],
            filter_labels
        )
