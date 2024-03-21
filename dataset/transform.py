import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (CastToTyped, SpatialPadd, RandCropByPosNegLabeld, Compose, CropForegroundd,
                              LoadImaged, RandFlipd, RandGaussianNoised, Spacingd,
                              RandGaussianSmoothd, Spacing, LoadImage,
                              RandScaleIntensityd, RandZoomd, ToTensord, EnsureTyped)
from monai.transforms import MapTransform


class ReadNumpyArray(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.to_tensor = ToTensord(keys=keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = np.load(data[key])
        return self.to_tensor(data)


class ClipNormalize(MapTransform):

    def __init__(self, keys, clip_values, normalize_values):
        super().__init__(keys)
        self.clip_values = clip_values
        self.normalize_values = normalize_values

    def __call__(self, data):
        for key in self.keys:
            data[key] = np.clip(data[key], self.clip_values[0], self.clip_values[1])
            data[key] = (data[key] - self.normalize_values[0]) / self.normalize_values[1]
        return data


def get_transforms(mode, cfg):
    # mode: train, val, test, selftrain
    if mode != "test":
        keys = ["image", "label"]
        spacing_mode = ("bilinear", "nearest")
        spacing_ac = [True, True]
        dtype = (np.float32, np.uint8)
    else:
        keys = ["image"]
        spacing_mode = "bilinear"
        spacing_ac = True
        dtype = (np.float32)

    # 1. load (4)
    load_transforms = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=False)
    ]

    # 2. sampling (4)
    sample_transforms = [
        Spacingd(keys=keys, pixdim=cfg.data.spacing, mode=spacing_mode, align_corners=spacing_ac),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=False),
        ClipNormalize(keys=['image'], clip_values=cfg.data.clip_values, normalize_values=cfg.data.normalize_values),
        SpatialPadd(keys=keys, spatial_size=cfg.data.patch_size),
        ToTensord(keys=keys)
    ]

    # 3. spatial transforms (9)
    if mode == "train":
        augmentation = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.data.patch_size,
                pos=cfg.data.pos_sample_num,
                neg=cfg.data.neg_sample_num,
                num_samples=cfg.data.num_samples,
            ),
            SpatialPadd(keys=keys, spatial_size=cfg.data.patch_size),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.8,
                max_zoom=1.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.2,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
        ]
    else:
        augmentation = []

    data_casting = [
        CastToTyped(keys=keys, dtype=dtype),
        EnsureTyped(keys=["image"])
    ]

    all_transforms = load_transforms + sample_transforms + augmentation + data_casting

    if mode == 'selftrain':
        all_transforms = [ReadNumpyArray(keys=keys)] + augmentation + data_casting

    return Compose(all_transforms)


def post_transform(_label, cfg, data):
    transform = Spacing(
        pixdim=cfg.data.spacing,
        mode="nearest",
        align_corners=True
    )

    load_transform = LoadImage(reader='NibabelReader', ensure_channel_first=True, image_only=False)
    img, img_meta = load_transform(data['image_meta_dict']['filename_or_obj'])
    label = MetaTensor(x=torch.zeros_like(img, dtype=torch.uint8), meta=img.meta)
    tr_label = transform(label)

    fg_start = data['foreground_start_coord']
    fg_end = data['foreground_end_coord']

    try:
        assert _label.shape[1:] == (fg_end[0] - fg_start[0], fg_end[1] - fg_start[1], fg_end[2] - fg_start[2]), \
            f"Label shape {_label.shape} does not match with foreground shape {fg_end[0] - fg_start[0], fg_end[1] - fg_start[1], fg_end[2] - fg_start[2]}"

        tr_label[..., fg_start[0]:fg_end[0], fg_start[1]:fg_end[1], fg_start[2]:fg_end[2]] = _label
        return transform.inverse(tr_label)
    except AssertionError as e:
        return None
