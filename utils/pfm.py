# Taken from https://gist.github.com/chpatrick/8935738
# Modified to work with Python3.
import numpy as np
import re
import sys


def load_pfm(file_path):
    """
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    :param file_path: Str.
    :return: Np array.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape), scale


def save_pfm(file_path, image, scale=1):
    """
    Save a Numpy array to a PFM file.
    :param file_path: Str.
    :param image: Np array.
    :param scale: Float.
    :return: Nothing.
    """
    with open(file_path, 'w') as file:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file.write('PF\n' if color else 'Pf\n')
        file.write('%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file.write('%f\n' % scale)

        image.tofile(file)
