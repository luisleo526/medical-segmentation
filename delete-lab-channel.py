from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing import Pool
from uuid import uuid4

import nibabel as nib
import numpy as np


def parse_args():
    parser = ArgumentParser(description='Delete the 1st label channel of the MSD dataset')
    parser.add_argument('-paths', nargs='+', help='Paths to the label files to process')
    parser.add_argument('-data_root', default='MergeTumor', help='Root directory to save the processed files')
    parser.add_argument('-threads', type=int, default=16, help='Number of threads to use')
    return parser.parse_args()


def process_image(label_path, data_root='MergeTumor'):
    try:
        nii_img = nib.load(label_path)
        nii_lab = nib.load(label_path.replace('labelsTr', 'imagesTr'))
    except Exception as e:
        nii_img = None
        nii_lab = None
        print(f'Error processing {label_path}: {e}')

    if nii_img is not None and nii_lab is not None:
        # Get the data as a numpy array
        data = nii_img.get_fdata()

        # Modify the data: set class 1 to 0 and class 2 to 1
        data[data == 1] = 0
        data[data == 2] = 1

        # Process data if '1' is present

        r = data[data == 1].size / data.size
        if r < 0.005:
            print(f"Skip {label_path}, ratio: {r}")
            return

        assert np.unique(data).size == 2, f"Wrong number of classes: {np.unique(data)}"

        # Create a new NIfTI image with the modified data
        new_nii_img = nib.Nifti1Image(data, affine=nii_img.affine, header=nii_img.header)

        filename = str(uuid4()) + '.nii.gz'

        # Save the modified image to a new file
        nib.save(new_nii_img, f'{data_root}/labelTr/{filename}')

        # Copy the corresponding image file to the new directory, by load and save
        nib.save(nii_lab, f'{data_root}/imageTr/{filename}')


if __name__ == '__main__':
    args = parse_args()
    data = []
    for root in args.paths:
        data += glob(f'{root}/*.nii.gz')

    print("Processing", len(data), "files")

    # Multiple processes
    with Pool(args.threads) as p:
        p.map(partial(process_image, data_root=args.data_root), data)
