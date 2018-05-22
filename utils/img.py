# To resolve some issues with MacOS and matplotlib:
# https://stackoverflow.com/questions/2512225/matplotlib-plots-not-showing-up-in-mac-osx
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(img):
    """
    Opens a window displaying the image.
    :param img: Numpy array of shape (Height, Width, Channels)
    :return: Nothing.
    """
    if img.dtype == 'float32' or img.dtype == 'float64':
        img = np.clip(img, 0.0, 1.0)
    plt.imshow(img)
    plt.show()


def read_image(img_path, as_float=False):
    """
    :param img_path: Str.
    :param as_float: Bool. If true, then return the image as floats between [0, 1] instead of uint8s between [0, 255].
    :return:
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        return img.astype(dtype=np.float32) / 255.0
    else:
        return img
