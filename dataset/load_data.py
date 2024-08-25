from glob import glob
from pathlib import Path


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

    label_parent_path = Path(cfg.data.train.labels).parent
    # datalist['train'] = [{'image': x, 'label': str(label_path / Path(x).name)} for x in images_tr]
    # datalist['val'] = [{'image': x, 'label': str(label_path / Path(x).name)} for x in images_val]

    for key, data_list in [('train', images_tr), ('val', images_val)]:
        for image_path in data_list:
            image_filename = Path(image_path).name
            label_filename = image_filename.replace(cfg.data.train.image_extension, cfg.data.train.label_extension)
            label_path = label_parent_path / label_filename

            datalist[key].append({'image': image_path, 'label': str(label_path)})

    datalist['test'] = [{'image': x} for x in images_ts]

    if cfg.debug:
        datalist['train'] = datalist['train'][:20]
        datalist['val'] = datalist['val'][:20]
        datalist['test'] = datalist['test'][:20]

    return datalist
