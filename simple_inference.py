from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from monai.data import NibabelWriter
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from tqdm import tqdm
from uvw import RectilinearGrid, DataArray
from monailabel.transform.writer import write_seg_nrrd
from monai.networks.utils import one_hot
from dataset import get_transforms, post_transform
from utils import initiate, extract_elements

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--artifact', type=str, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--overlap', type=float, default=0.7)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    images = glob(str(args.data / '*.nii.gz'))
    args.output.mkdir(exist_ok=True, parents=True)

    api = wandb.Api()

    artifact = api.artifact(args.artifact, type="model")
    artifact_dir = artifact.download()
    cfg = OmegaConf.create(artifact.metadata)

    mean, std = cfg.data.normalize_values

    model = initiate(cfg.model, cfg=cfg, skip=True)
    model.load_state_dict(torch.load(artifact_dir + '/pytorch_model.bin'))
    model.cuda()
    model.eval()

    transform = get_transforms('test', cfg)

    for image in tqdm(images):
        instance = Path(image).stem.split('.')[0]
        Path(args.output / instance).mkdir(exist_ok=True, parents=True)
        data = transform({'image': image})
        image_tensor = data['image'].unsqueeze(0).cuda()

        with torch.no_grad():
            pred = sliding_window_inference(inputs=image_tensor, roi_size=cfg.data.patch_size,
                                            sw_batch_size=args.batch_size, predictor=model, overlap=args.overlap)
            prob = torch.nn.functional.softmax(pred, dim=1).half()

            # Un-Batch
            p_label = pred.argmax(1, keepdim=True).squeeze(0).cpu()

            # Reverse the spatial transforms
            p_label_inv = post_transform(p_label, cfg, data)
            p_label_inv = one_hot(p_label_inv, len(cfg.data.targets), dim=0)
            p_label_inv = p_label_inv.cpu().numpy()

            if p_label_inv is not None:
                write_seg_nrrd(
                    p_label_inv,
                    f"{str(args.output)}/{instance}.seg.nrrd",
                    p_label_inv.dtype,
                    data['image_meta_dict']['affine'].numpy(),
                    cfg.data.targets
                )
