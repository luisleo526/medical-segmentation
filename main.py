from importlib import import_module

import hydra
from accelerate import Accelerator
from monai.data import ThreadDataLoader, CacheDataset
from omegaconf import DictConfig, OmegaConf

from dataset import load_datalist, get_transforms


def get_class(x):
    module = x[:x.rfind(".")]
    obj = x[x.rfind(".") + 1:]
    return getattr(import_module(module), obj)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    accelerator = Accelerator()

    datalist = load_datalist(cfg, accelerator.process_index, accelerator.num_processes)
    print(datalist['train'][0])
    datasets = {k: CacheDataset(data=v if not cfg.debug else v[:2], transform=get_transforms(k, cfg))
                for k, v in datalist.items()}
    dataloaders = {k: ThreadDataLoader(v, batch_size=cfg.model.batch_size[k], num_workers=cfg.num_workers,
                                       use_thread_workers=True, pin_memory=True)
                   for k, v in datasets.items()}

    dataloaders = {k: accelerator.prepare(dataloader, device_placement=[True]) for k, dataloader in dataloaders.items()}

    model = get_class(cfg.model.network.type)(cfg)
    model = accelerator.prepare(model, device_placement=[True])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
