import os

import torch
from monai.networks.nets import SwinUNETR


class SegNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_classes = len(cfg.data.targets)

        self.model = SwinUNETR(
            img_size=cfg.data.patch_size,
            in_channels=cfg.data.in_channels,
            out_channels=num_classes,
            spatial_dims=3,
            **cfg.model.network.params
        )

        head_ckpt = f"model/swinvit_{cfg.model.network.params.feature_size}.pth"
        if cfg.model.network.load_head and os.path.exists(head_ckpt):
            self.model.swinViT.load_state_dict(torch.load(head_ckpt))
