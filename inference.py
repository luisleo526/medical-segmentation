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

from dataset import get_transforms, post_transform
from utils import initiate, extract_elements


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--artifact', type=str, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--overlap', type=float, default=0.7)
    parser.add_argument('--store_raw', action='store_true')
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
    model.load_state_dict(torch.load(artifact_dir + '/pytorch_model.bin'), strict=False)
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
            p_label = p_label.squeeze(0).numpy()

            if p_label_inv is not None:
                # Save label as NIfTI using NibabelWriter
                writer = NibabelWriter()
                writer.set_data_array(p_label_inv, channel_dim=0)
                writer.set_metadata(data['image_meta_dict'])
                filename = instance + ".nii.gz"
                filename = args.output / instance / filename
                writer.write(filename)

                # store raw prediction
                if args.store_raw:
                    filepath = args.output / instance / 'raw_prediction.pt'
                    torch.save(prob, filepath)

            uniques = [i for i in range(p_label.shape[-1]) if np.unique(p_label[..., i]).size > 1]
            start_idx = min(uniques)
            end_idx = max(uniques)

            if end_idx - start_idx > args.samples:
                for z in extract_elements(range(start_idx, end_idx + 1), args.samples):
                    img = image_tensor.squeeze(0).squeeze(0).cpu().numpy()[..., z]
                    H, W = img.shape
                    img = np.stack([img, img, img], axis=-1)
                    img = (img * std + mean)
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)

                    lab = p_label[..., z]

                    img = overlay(img, lab == 1, (255, 0, 0), 0.5)
                    img = overlay(img, lab == 2, (0, 255, 0), 0.5)

                    filename = f"2d_overlap_{z}.png"
                    filename = args.output / instance / filename
                    cv2.imwrite(str(filename), img)

        nx, ny, nz = p_label.shape

        x = np.linspace(0, nx, nx)
        y = np.linspace(0, ny, ny)
        z = np.linspace(0, nz, nz)

        filename = "3D_mask.vtr"
        filename = args.output / instance / filename
        grid = RectilinearGrid(str(filename), (x, y, z), compression=True)
        grid.addPointData(DataArray(p_label, range(3), 'mask'))
        grid.write()
