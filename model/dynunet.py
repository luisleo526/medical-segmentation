import torch
from monai.networks.nets import DynUNet

from utils import initiate, get_weights


def get_kernels_strides(cfg):
    sizes, spacings = cfg.data.patch_size, cfg.data.spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


class SegNet(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        num_classes = len(cfg.data.targets)
        kernels, strides = get_kernels_strides(cfg)
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=cfg.data.in_channels,
            out_channels=num_classes,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
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
        pred = torch.unbind(y_pred, dim=1)
        loss = sum([0.5 ** i * self.loss_fn(p, y_true) for i, p in enumerate(pred)])
        return pred[0], loss

    def forward(self, x, y=None):
        if type(x) is dict:
            return self.compute_loss(self.model(x['image']), x['label'])
        elif torch.is_tensor(x):
            if torch.is_tensor(y):
                return self.loss_fn(x, y)
            else:
                return self.model(x)
