# To resolve some issues with MacOS and matplotlib:
# https://stackoverflow.com/questions/2512225/matplotlib-plots-not-showing-up-in-mac-osx
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


def tf_random_crop(images, crop_size):
    """
    Picks a random crop region based on crop_size, and applies the same crop rect to all images.
    :param images: List of image tensors. Number of channels does not matter. Shape is (H, W, C).
    :param crop_size: Tuple of (int (H), int (W)). Size to crop the training examples to before feeding to network.
                      If None, then no cropping will be performed.
    :return: List of image tensors
    """
    if crop_size is not None and len(images) > 0:
        crop_height = crop_size[0]
        crop_width = crop_size[1]
        assert crop_height > 0 and crop_width > 0
        shape = tf.shape(images[0])
        H = shape[0]
        W = shape[1]
        # Pick out the crop region location.
        rand_y_start = tf.random_uniform((), 0, H - crop_height + 1, dtype=tf.int32)
        rand_x_start = tf.random_uniform((), 0, W - crop_width + 1, dtype=tf.int32)
        rand_y_end = rand_y_start + crop_height
        rand_x_end = rand_x_start + crop_width
        # Do cropping.
        cropped_images = []
        for image in images:
            cropped_images.append(image[rand_y_start:rand_y_end, rand_x_start:rand_x_end, :])
        return cropped_images
    return images


def tf_image_augmentation(images, config):
    """
    Does a set of basic image augmentations to a list of images.
    :param images: List of image tensors. Number of channels does not matter. Shape is (H, W, C).
    :param config: Dict.
    :return: List of image tensors
    """
    if len(images) > 0:
        shape = tf.shape(images[0])
        H = shape[-3]
        W = shape[-2]
        C = shape[-1]

        # Contrast.
        rand_constrast = tf.random_uniform((), config['contrast_min'], config['contrast_max'], dtype=tf.float32)
        # Gamma and gain.
        rand_gamma = tf.random_uniform((), config['gamma_min'], config['gamma_max'], dtype=tf.float32)
        rand_gain = tf.random_uniform((), config['gain_min'], config['gain_max'], dtype=tf.float32)
        # Brightness.
        rand_brightness = tf.random_normal((), mean=0.0, stddev=config['brightness_stddev'], dtype=tf.float32)
        # Hue.
        rand_hue = tf.random_uniform((), config['hue_min'], config['hue_max'], dtype=tf.float32)
        # Noise.
        rand_sigma = tf.random_uniform((), 0.0, config['noise_stddev'], dtype=tf.float32)

        randomized_images = []
        for image in images:
            new_image = (image ** rand_gamma) * rand_gain
            new_image = tf.image.adjust_contrast(new_image, rand_constrast)
            new_image = tf.image.adjust_hue(new_image, rand_hue)

            # Gaussian noise is created per image.
            rand_image = tf.random_normal((H, W, C), mean=rand_brightness, stddev=rand_sigma, dtype=tf.float32)
            new_image = new_image + rand_image

            randomized_images.append(new_image)
        return randomized_images
    return images
