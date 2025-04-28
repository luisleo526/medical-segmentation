import torch
from monai.networks.nets import AttentionUnet

from utils import initiate, get_weights


class SegNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=cfg.data.in_channels,
            out_channels=len(cfg.data.targets),
            **cfg.model.params
        )

#         weights = []
#         for target in cfg.data.targets:
#             weights.append(get_weights(target))

#         if sum(weights) == 0:
#             weights = None
#         else:
#             weights_max = max(weights)
#             weights = [w / weights_max for w in weights]
#             weights = torch.tensor(weights).float()
            
        weights = None

        self.loss_fn = initiate(cfg.loss_fn, weight=weights)

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
