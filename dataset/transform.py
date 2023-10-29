import numpy as np
from monai.transforms import (CastToTyped, SpatialPadd, RandCropByPosNegLabeld, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged,
                              RandFlipd, RandGaussianNoised, Spacingd, RandGaussianSmoothd,
                              RandScaleIntensityd, ScaleIntensityRanged, NormalizeIntensityd,
                              RandZoomd, ToTensord, EnsureTyped, SelectItemsd)
from monai.transforms import MapTransform


class ReadNumpyArray(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.to_tensor = ToTensord(keys=keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = np.load(data[key])
        return self.to_tensor(data)


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

    load_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]

    a_min = (cfg.data.clip_values[0] - cfg.data.normalize_values[0]) / cfg.data.normalize_values[1]
    a_max = (cfg.data.clip_values[1] - cfg.data.normalize_values[0]) / cfg.data.normalize_values[1]

    # 2. sampling
    sample_transforms = [
        Spacingd(keys=keys, pixdim=cfg.data.spacing, mode=spacing_mode, align_corners=spacing_ac),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=False),
        NormalizeIntensityd(keys="image",
                            subtrahend=cfg.data.normalize_values[0],
                            divisor=cfg.data.normalize_values[1]),
        ScaleIntensityRanged(keys="image",
                             a_min=a_min, a_max=a_max,
                             b_min=0, b_max=1, clip=True),
        ToTensord(keys=keys),
    ]

    # 3. spatial transforms
    if mode == "train" or mode == "selftrain":
        other_transforms = [
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.data.patch_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.data.patch_size,
                pos=cfg.data.pos_sample_num,
                neg=cfg.data.neg_sample_num,
                num_samples=cfg.data.num_samples,
                image_key="image",
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.data.patch_size),
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
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
            SelectItemsd(keys=["image", "label"])
        ]
    elif mode == "val":
        other_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
            SelectItemsd(keys=["image", "label"])
        ]
    else:
        other_transforms = [
            CastToTyped(keys=["image"], dtype=(np.float32)),
            EnsureTyped(keys=["image"]),
            SelectItemsd(keys=["image"])
        ]

    all_transforms = load_transforms + sample_transforms + other_transforms

    if mode == 'selftrain':
        all_transforms = [ReadNumpyArray(keys=keys)] + other_transforms

    return Compose(all_transforms)
