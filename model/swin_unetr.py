import os

import torch
from monai.losses import DiceCELoss
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
            **cfg.model.params
        )

        head_ckpt = f"model/swinvit_{cfg.model.network.params.feature_size}.pth"

        if cfg.model.network.load_head and os.path.exists(head_ckpt):
            self.model.swinViT.load_state_dict(torch.load(head_ckpt))

        self.loss_fn = DiceCELoss(include_background=False, softmax=True, reduction='mean', to_onehot_y=True)

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        return y_pred, loss

    def forward(self, x, y=None):
        if type(x) is dict:
            return self.compute_loss(self.model(x['image']), x['label'])
        elif torch.is_tensor(x):
            if torch.is_tensor(y):
                return self.loss_fn(x, y)
            else:
                return self.model(x)
