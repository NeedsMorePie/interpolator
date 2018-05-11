# Mostly copied from https://github.com/nameless-Chatoyant/PWC-Net_pytorch/blob/master/modules.py.
# Master branch commit 2225ad2082371126cc9c8e57a8b962a88933a8c0.

import tensorflow as tf


def cost_volume(c1, c2, search_range=4):
    """
    See https://arxiv.org/pdf/1709.02371.pdf.
    For each pixel in c1, we will compute correlations with its spatial neighbors in c2.
    :param c1: Input tensor, with shape (batch, width, height, features).
    :param c2: Input tensor with the exact same shape as c1.
    :param search_range: The search square's side length = 2 * search_range + 1.
    :return: A tensor with shape (batch, width, height, s * s), where s is each search square's side length.
    """
    square_len = 2 * search_range + 1
    square_area = square_len ** 2
    cv_shape = tf.shape(c1)
    cv_shape[-1] = square_area
    cv = tf.zeros(cv_shape)

    # This is pretty smart.
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

            k = square_len * i + j
            costs = tf.reduce_sum(c1[:, slice_h, slice_w, :] * c2[:, slice_h_r, slice_w_r, :], axis=-1)
            cv[:, slice_h, slice_w, k] = costs

    return cv
