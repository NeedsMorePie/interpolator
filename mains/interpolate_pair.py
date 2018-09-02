import argparse
import numpy as np
from common.utils import img
from pwcnet import model
from interp.interp import Interp
import tensorflow as tf
from tensorflow.contrib import predictor


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    # img_0 = img.read_image(args.image_0)
    # img_1 = img.read_image(args.image_1)
    img_0 = np.zeros((1, 256, 256, 3))
    img_1 = np.zeros((1, 256, 256, 3))
    interpolator = Interp(saved_model_dir=args.saved_model_dir)
    print('Loading the SavedModel from: ', args.saved_model_dir)
    interpolator.load_saved_model()
    preds = interpolator.predict(img_0, img_1)
    print(preds)
    print(np.mean(preds))
    print(np.shape(preds))


def add_args(parser):
    parser.add_argument('-i', '--image_0', type=str,
                        help='Image at t=0.')
    parser.add_argument('-j', '--image_1', type=str,
                        help='Image at t=1.')
    parser.add_argument('-s', '--saved_model_dir', type=str,
                        help='Path to the SavedModel that we will load.')
    parser.add_argument('-o', '--out_directory', type=str,
                        help='Directory that we output into.')


if __name__ == '__main__':
    main()