# Mostly copied from https://github.com/nameless-Chatoyant/PWC-Net_pytorch/blob/master/modules.py.
# Master branch commit 2225ad2082371126cc9c8e57a8b962a88933a8c0.

import tensorflow as tf


def cost_volume(c1, c2, search_range=4):
    """
    See https://arxiv.org/pdf/1709.02371.pdf.
    For each pixel in c1, we will compute correlations with its spatial neighbors in c2.
    :param c1: Input tensor, with shape (batch, height, width, features).
    :param c2: Input tensor with the exact same shape as c1.
    :param search_range: The search square's side length = 2 * search_range + 1.
    :return: A tensor with shape (batch, height, width, s * s), where s is each search square's side length.
    """
    square_len = 2 * search_range + 1

    # cv is cost volume, not OpenCV.
    square_area = square_len ** 2
    cv_shape = tf.shape(c1)[:-1]
    cv_shape = tf.concat([cv_shape, [square_area]], axis=0)
    cv = tf.zeros(cv_shape)

    # Form an index matrix to help us update sparsely later on.
    cv_height, cv_width = cv_shape[1], cv_shape[2]
    x_1d, y_1d = tf.range(0, cv_width), tf.range(0, cv_height)
    x_2d, y_2d = tf.meshgrid(x_1d, y_1d)
    indices_2d = tf.stack([y_2d, x_2d], axis=-1)

    # This is pretty smart.
    cv_slices = []
    for i in range(-search_range, search_range + 1):
        for j in range(-search_range, search_range + 1):

            # Note that 'Python'[slice(None)] returns 'Python'.
            if i < 0:
                slice_h, slice_h_r = slice(None, i), slice(-i, None)
            elif i > 0:
                slice_h, slice_h_r = slice(i, None), slice(None, -i)
            else:
                slice_h, slice_h_r = slice(None), slice(None)

            if j < 0:
                slice_w, slice_w_r = slice(None, j), slice(-j, None)
            elif j > 0:
                slice_w, slice_w_r = slice(j, None), slice(None, -j)
            else:
                slice_w, slice_w_r = slice(None), slice(None)

            costs = tf.reduce_mean(c1[:, slice_h, slice_w, :] * c2[:, slice_h_r, slice_w_r, :], axis=-1)

            # Get the coordinates for scatter update, where each element is an (y, x, z) coordinate.
            cur_indices = indices_2d[slice_h, slice_w]
            cur_indices = tf.reshape(cur_indices, (-1, 2))

            # The batch dimension needs to be moved to the end to make slicing work correctly.
            costs = tf.reshape(costs, (tf.shape(costs)[0], -1))
            costs = tf.transpose(costs, [1, 0])
            batch_dim = tf.shape(c1)[0]
            target_shape = [cv_height, cv_width, batch_dim]
            cv_slice = tf.scatter_nd(cur_indices, costs, target_shape)
            cv_slices.append(cv_slice)

    cv = tf.stack(cv_slices, axis=-1)
    cv = tf.transpose(cv, [2, 0, 1, 3])
    return cv