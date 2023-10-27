from glob import glob
from pathlib import Path

from monai.data.utils import partition_dataset


def load_datalist(cfg, process_id, num_process):
    images_tr = glob(cfg.data.train.images)
    images_ts = glob(cfg.data.test.images)

    # Same base name for images and labels
    labels_tr = glob(cfg.data.train.labels)

    assert len(images_tr) > 0, "No training images found"
    assert len(images_ts) > 0, "No test images found"
    assert len(images_tr) == len(labels_tr), "Number of images and labels must be the same"

    print(len( images_tr), len(images_ts), len(labels_tr))

    images_tr = sorted(images_tr)
    images_ts = sorted(images_ts)

    num_of_train = int(len(images_tr) * 0.8)

    images_tr = images_tr[:num_of_train]
    images_val = images_tr[num_of_train:]

    if num_process > 1:
        images_tr = partition_dataset(images_tr, num_partitions=num_process, even_divisible=True)[process_id]
        images_val = partition_dataset(images_val, num_partitions=num_process, even_divisible=True)[process_id]
        images_ts = partition_dataset(images_ts, num_partitions=num_process, even_divisible=True)[process_id]

    datalist = {'train': [], 'val': [], 'test': []}

    label_path = Path(cfg.data.train.labels).parent
    datalist['train'] = [{'image': x, 'label': str(label_path / Path(x).name)} for x in images_tr]
    datalist['val'] = [{'image': x, 'label': str(label_path / Path(x).name)} for x in images_val]
    datalist['test'] = [{'image': x} for x in images_ts]

    return datalist
