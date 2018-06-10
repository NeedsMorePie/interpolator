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


FLOW_CHANNELS = 2
FLOW_TAG_FLOAT = 202021.25


def read_flow_file(file_name):
    """
    :param file_name: str.
    :return: Numpy array of shape (height, width, FLOW_CHANNELS).
        Returns None if the tag in the file header was invalid.
    """
    with open(file_name, 'rb') as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if tag != FLOW_TAG_FLOAT:
            return None
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]

        num_image_floats = width * height * FLOW_CHANNELS
        image = np.fromfile(f, dtype=np.float32, count=num_image_floats)
        image = image.reshape((height, width, FLOW_CHANNELS))

        return image


def get_flow_visualization(flow):
    """
    Uses the HSV color wheel for the visualization.
    Red means movement to the right, blue means movement to the left, yellow means movement downward,
    purple means movement upward.
    :param flow: Numpy array of shape (Height, Width, 2).
    :return: Rgb uint8 image.
    """
    abs_image = np.abs(flow)
    flow_mean = np.mean(abs_image)
    flow_std = np.std(abs_image)

    # Apply some kind of normalization. Divide by the perceived maximum (mean + std)
    flow = flow / (flow_mean + flow_std + 1e-10)

    # Get the angle and radius.
    # sqrt(x^2 + y^2).
    radius = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
    radius_clipped = np.clip(radius, 0.0, 1)
    # Arctan2(-y, -x) / pi so that it's between [-1, 1].
    angle = np.arctan2(-flow[..., 1], -flow[..., 0]) / np.pi

    # Hue is between [0, 1] but scaled by (179.0 / 255.0) because hue is usually between [0, 179].
    hue = np.clip((angle + 1.0) / 2.0, 0.0, 1.0) * (179.0 / 255.0)
    # Saturation is always 0.75 (it's more aesthetic).
    saturation = np.ones(shape=hue.shape, dtype=np.float) * 0.75
    # Value (brightness), is the radius (magnitude) of the flow.
    value = radius_clipped

    hsv_img = (np.clip(np.stack([hue, saturation, value], axis=-1) * 255, 0, 255)).astype(dtype=np.uint8)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


def get_tf_flow_visualization(flow):
    """
    :param flow: Optical flow tensor.
    :return: RGB image normalized between 0 and 1.
    """
    with tf.name_scope('flow_visualization'):
        # B, H, W, C dimensions.
        abs_image = tf.abs(flow)
        flow_mean, flow_var = tf.nn.moments(abs_image, axes=[1, 2, 3])
        flow_std = tf.sqrt(flow_var)

        # Apply some kind of normalization. Divide by the perceived maximum (mean + std)
        flow = flow / tf.expand_dims(tf.expand_dims(
            tf.expand_dims(flow_mean + flow_std + 1e-10, axis=-1), axis=-1), axis=-1)

        radius = tf.sqrt(tf.reduce_sum(tf.square(flow), axis=-1))
        radius_clipped = tf.clip_by_value(radius, 0.0, 1.0)
        angle = tf.atan2(-flow[..., 1], -flow[..., 0]) / np.pi

        hue = tf.clip_by_value((angle + 1.0) / 2.0, 0.0, 1.0)
        saturation = tf.ones(shape=tf.shape(hue), dtype=tf.float32) * 0.75
        value = radius_clipped
        hsv = tf.stack([hue, saturation, value], axis=-1)
        return tf.image.hsv_to_rgb(hsv)


def show_flow_image(flow):
    """
    Opens a window displaying an optical flow visualization.
    :param flow: Numpy array of shape (Height, Width, 2).
    :return: Nothing.
    """
    plt.imshow(get_flow_visualization(flow))
    plt.show()


def tf_flip_flow(flow, images, left_right, up_down):
    """
    :param flow: Optical flow tensor. Shape is (H, W, C).
    :param images: List of image tensors.
    :param left_right: Whether to flip left/right.
    :param up_down: Whether to flip up/down.
    :return: new_flow (tensor), new_images (list of tensors).
    """
    # When reversed, the flow vector needs to be reversed too.
    if flow is not None:
        new_flow = tf.cond(left_right, lambda: tf.reverse(flow * tf.constant([[[-1.0, 1.0]]]), [1]), lambda: flow)
        new_flow = tf.cond(up_down, lambda: tf.reverse(new_flow * tf.constant([[[1.0, -1.0]]]), [0]), lambda: new_flow)
    else:
        new_flow = None

    new_images = []
    for image in images:
        new_image = tf.cond(left_right, lambda: tf.reverse(image, [1]), lambda: image)
        new_image = tf.cond(up_down, lambda: tf.reverse(new_image, [0]), lambda: new_image)
        new_images.append(new_image)

    return new_flow, new_images


def tf_random_flip_flow(flow, images):
    """
    Randomly flips a flow and a set of corresponding images in unison.
    :param flow: Optical flow tensor. Shape is (H, W, C).
    :param images: List of image tensors.
    :return: new_flow (tensor), new_images (list of tensors).
    """
    left_right_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
    up_down_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
    return tf_flip_flow(flow, images, left_right_cond, up_down_cond)


def tf_scale_flow(flow, images, scale):
    """
    :param flow: Optical flow tensor. Shape is (H, W, C).
    :param images: List of image tensors.
    :param scale: How much to scale by. >1 means vertical scale, <1 means horizontal scale.
    :return: new_flow (tensor), new_images (list of tensors).
    """
    shape = tf.shape(flow)
    H = shape[0]
    H_f = tf.cast(H, dtype=tf.float32)
    W = shape[1]
    W_f = tf.cast(W, dtype=tf.float32)

    scale_vert_cond = tf.greater(scale, 1.0)

    # When scaled, the flow vector needs to be scaled too.
    # Notice that horizontal scaling is just 1/scale when scale < 1.
    if flow is not None:
        new_flow = tf.cond(scale_vert_cond,
                           lambda: tf.convert_to_tensor([[[1.0, scale]]]) * tf.image.resize_image_with_crop_or_pad(
                               tf.image.resize_images(flow, [tf.cast(H_f * scale, tf.int32), W]), H, W),
                           lambda: tf.convert_to_tensor(
                               [[[1.0 / scale, 1.0]]]) * tf.image.resize_image_with_crop_or_pad(
                               tf.image.resize_images(flow, [H, tf.cast(W_f / scale, tf.int32)]), H, W))
    else:
        new_flow = None

    new_images = []
    for image in images:
        new_image = tf.cond(scale_vert_cond,
                            lambda: tf.image.resize_image_with_crop_or_pad(
                                tf.image.resize_images(image, [tf.cast(H_f * scale, tf.int32), W]), H, W),
                            lambda: tf.image.resize_image_with_crop_or_pad(
                                tf.image.resize_images(image, [H, tf.cast(W_f / scale, tf.int32)]), H, W))
        new_images.append(new_image)

    return new_flow, new_images


def tf_random_scale_flow(flow, images, config):
    """
    Randomly scales a flow and a set of corresponding images in unison.
    :param flow: Optical flow tensor. Shape is (H, W, C).
    :param images: List of image tensors.
    :param config: Dict.
    :return: new_flow (tensor), new_images (list of tensors).
    """
    scale = tf.random_uniform((), config['scale_min'], config['scale_max'], dtype=tf.float32)
    return tf_scale_flow(flow, images, scale)
