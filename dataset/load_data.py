from glob import glob
from pathlib import Path

from monai.data.utils import partition_dataset


def load_datalist(cfg):
    root = cfg.data.root

    cfg.data.train.images = str(Path(root) / Path(cfg.data.train.images))
    cfg.data.train.labels = str(Path(root) / Path(cfg.data.train.labels))
    cfg.data.test.images = str(Path(root) / Path(cfg.data.test.images))

    images_tr = glob(cfg.data.train.images)
    images_ts = glob(cfg.data.test.images)

    # Same base name for images and labels
    labels_tr = glob(cfg.data.train.labels)

    assert len(images_tr) > 0, "No training images found"
    assert len(images_ts) > 0, "No test images found"
    assert len(images_tr) == len(labels_tr), "Number of images and labels must be the same"

    images_tr = sorted(images_tr)
    images_ts = sorted(images_ts)

    num_of_train = int(len(images_tr) * 0.8)
    images_tr, images_val = images_tr[:num_of_train], images_tr[num_of_train:]

    datalist = {'train': [], 'val': [], 'test': []}

    label_path = Path(cfg.data.train.labels).parent
    datalist['train'] = [{'image': x, 'label': str(label_path / Path(x).name)} for x in images_tr]
    datalist['val'] = [{'image': x, 'label': str(label_path / Path(x).name)} for x in images_val]
    datalist['test'] = [{'image': x} for x in images_ts]

    return datalist
