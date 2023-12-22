import numpy as np
from monai.transforms import (CastToTyped, SpatialPadd, RandCropByPosNegLabeld, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, RandFlipd, RandGaussianNoised, Spacingd,
                              RandGaussianSmoothd,
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
    else:
        keys = ["image"]
        spacing_mode = "bilinear"
        spacing_ac = True

    # 1. load (4)
    load_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys)
    ]

    # 2. sampling (4)
    sample_transforms = [
        Spacingd(keys=keys, pixdim=cfg.data.spacing, mode=spacing_mode, align_corners=spacing_ac),
        SpatialPadd(keys=keys, spatial_size=cfg.data.patch_size),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=False),
        ClipNormalize(keys=['image'], clip_values=cfg.data.clip_values, normalize_values=cfg.data.normalize_values),
        ToTensord(keys=keys),
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
                image_key="image",
            ),
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

    # 4. data casting (2)
    if mode == 'test':
        data_casting = [
            CastToTyped(keys=["image"], dtype=(np.float32)),
            EnsureTyped(keys=["image"])
        ]
    else:
        data_casting = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"])
        ]

    all_transforms = load_transforms + sample_transforms + augmentation + data_casting

    if mode == 'selftrain':
        all_transforms = [ReadNumpyArray(keys=keys)] + augmentation + data_casting

    return Compose(all_transforms)
