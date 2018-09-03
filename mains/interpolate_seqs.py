import argparse
import numpy as np
import os
import glob
from common.utils import img
from common.utils import data
from pwcnet import model # Needed for defining ops (e.g Correlation).
from interp.interp import Interp
from PIL import Image


def main():
    """
    Images in the same shot should have the same name, followed by their index.
    For example, the input directory may look like this:
        foo_0.jpg
        foo_1.jpg
        foo_2.jpg
        lmao_0.jpg
        lmao_1.jpg
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    file_names = glob.glob(os.path.join(args.input_directory, '*'))
    groups = {}
    for name in file_names:
        group, idx = data.get_group_and_idx(name)
        if group is None:
            continue
        if group in groups:
            groups[group].append(idx)
        else:
            groups[group] = [idx]

    for key, value in groups.items():
        groups[key] = groups[key].sort()

    img_0 = np.zeros((1, 256, 256, 3))
    img_1 = np.zeros((1, 256, 256, 3))
    interpolator = Interp(saved_model_dir=args.saved_model_dir)
    print('Loading the SavedModel from: ', args.saved_model_dir)

    # TODO: Iterate through groups in batches for interpolation and image saving.

    # Interpolate and save the image.
    interpolated = interpolator.interpolate_from_saved_model(img_0, img_1)
    im = Image.fromarray(interpolated)
    im.save(args.out_name)


def add_args(parser):
    parser.add_argument('-d', '--input_directory', type=str,
                        help='Input directory.')
    parser.add_argument('-s', '--saved_model_dir', type=str,
                        help='Path to the SavedModel that we will load.')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Maximum batch size that we run the model with.')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='Directory that we output into.')


if __name__ == '__main__':
    main()