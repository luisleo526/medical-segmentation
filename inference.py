from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from uvw import RectilinearGrid, DataArray

import wandb
from dataset import get_transforms
from utils import initiate


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--artifact', type=str, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    images = glob(args.data / '*.nii.gz')
    args.output.mkdir(exist_ok=True, parents=True)

    api = wandb.Api()

    artifact = api.artifact(args.artifact, type="model")
    artifact_dir = artifact.download()
    cfg = artifact.metadata

    model = initiate(cfg.model.network, cfg=cfg, skip=True)
    model.load_state_dict(torch.load(artifact_dir + '/pytorch_model.bin'))
    model.cuda()
    model.eval()

    transform = get_transforms('test', cfg, torch.device('cuda'))

    for image in tqdm(images):
        image_tensor = transform({'image': image})['image'].unsqueeze(0).cuda()

        with torch.no_grad():
            pred = sliding_window_inference(inputs=image_tensor, roi_size=cfg.data.patch_size,
                                            sw_batch_size=args.batch_size, predictor=model, overlap=0.7)
            p_label = pred.argmax(1, keepdim=True).squeeze(0).squeeze(0).cpu().numpy()

        nx, ny, nz = p_label.shape

        x = np.linspace(0, nx, nx)
        y = np.linspace(0, ny, ny)
        z = np.linspace(0, nz, nz)

        output_dir = args.output / Path(image).stem
        output_dir = output_dir.to_string().split('.')[0] + '.vtk'
        grid = RectilinearGrid(output_dir, (x, y, z), compression=True)
        grid.addPointData(DataArray(p_label, range(3), 'mask'))
        grid.write()
