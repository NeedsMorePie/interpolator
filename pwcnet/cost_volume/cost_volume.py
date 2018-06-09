# Mostly copied from https://github.com/nameless-Chatoyant/PWC-Net_pytorch/blob/master/modules.py.
# Master branch commit 2225ad2082371126cc9c8e57a8b962a88933a8c0.

import tensorflow as tf


def cost_volume(c1, c2, search_range=4):
    """
    See https://arxiv.org/pdf/1709.02371.pdf.
    For each pixel in c1, we will compute correlations with its spatial neighbors in c2.
    :param c1: Tensor. Feature map of shape [batch_size, H, W, num_features].
    :param c2: Input tensor with the exact same shape as c1.
    :param search_range: The search square's side length is equal to 2 * search_range + 1.
    :return: Tensor. Cost volume of shape [batch_size, H, W, s * s], where s is equal to 2 * search_range + 1.
    """
    with tf.name_scope('cost_volume'):
        square_len = 2 * search_range + 1
        square_area = square_len ** 2
        cv_shape = tf.concat([tf.shape(c1)[:-1], [square_area]], axis=0)
        c1_transposed = tf.transpose(c1, [1, 2, 3, 0])
        c2_transposed = tf.transpose(c2, [1, 2, 3, 0])

        # Form an index matrix to help us update sparsely later on.
        cv_height, cv_width = cv_shape[1], cv_shape[2]
        x_1d, y_1d, z_1d = tf.range(0, cv_width), tf.range(0, cv_height), tf.range(0, square_area)
        x_2d, y_2d, z_2d = tf.meshgrid(x_1d, y_1d, z_1d)
        indices_3d = tf.stack([y_2d, x_2d, z_2d], axis=-1)

        all_indices, all_costs = [], []
        c1_regions, c2_regions = [], []
        cur_z_index = square_area - 1
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

                # Get the coordinates for scatter update, where each element is a (y, x, z) coordinate.
                cur_indices = indices_3d[slice_h, slice_w, cur_z_index]
                cur_indices = tf.reshape(cur_indices, (-1, 3))

                # The batch dimension needs to be moved to the end to make slicing work correctly.
                costs = tf.reshape(costs, (tf.shape(costs)[0], -1))
                costs = tf.transpose(costs, [1, 0])
                all_costs.append(costs)
                all_indices.append(cur_indices)
                c1_regions.append(c1_transposed[slice_h, slice_w, :, :])
                c2_regions.append(c2_transposed[slice_h_r, slice_w_r, :, :])
                cur_z_index -= 1

        all_costs = tf.concat(all_costs[::-1], axis=0)
        all_indices = tf.concat(all_indices[::-1], axis=0)
        batch_dim = tf.shape(c1)[0]
        target_shape = [cv_height, cv_width, square_area, batch_dim]
        cv = tf.scatter_nd(all_indices, all_costs, target_shape)
        cv = tf.transpose(cv, [3, 0, 1, 2])
        return cv


def cost_volume_grouped_reduce(c1, c2, search_range=4):
    """
    See https://arxiv.org/pdf/1709.02371.pdf.
    For each pixel in c1, we will compute correlations with its spatial neighbors in c2.
    :param c1: Tensor. Feature map of shape [batch_size, H, W, num_features].
    :param c2: Input tensor with the exact same shape as c1.
    :param search_range: The search square's side length is equal to 2 * search_range + 1.
    :return: Tensor. Cost volume of shape [batch_size, H, W, s * s], where s is equal to 2 * search_range + 1.
    """
    with tf.name_scope('cost_volume'):
        square_len = 2 * search_range + 1
        square_area = square_len ** 2
        cv_shape = tf.concat([tf.shape(c1)[:-1], [square_area]], axis=0)

        # The batch dimension needs to be moved to the end to make slicing work correctly.
        c1_transposed = tf.transpose(c1, [1, 2, 3, 0])
        c2_transposed = tf.transpose(c2, [1, 2, 3, 0])

        # Form an index matrix to help us update sparsely later on.
        cv_height, cv_width = cv_shape[1], cv_shape[2]
        x_1d, y_1d, z_1d = tf.range(0, cv_width), tf.range(0, cv_height), tf.range(0, square_area)
        x_2d, y_2d, z_2d = tf.meshgrid(x_1d, y_1d, z_1d)
        indices_3d = tf.stack([y_2d, x_2d, z_2d], axis=-1)

        all_indices, all_costs = [], []
        all_c1_regions, all_c2_regions = [], []
        cur_z_index = square_area - 1
        num_channels = tf.shape(c1)[-1]
        batch_size = tf.shape(c1)[0]
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

                # Get the coordinates for scatter update, where each element is a (y, x, z) coordinate.
                cur_indices = indices_3d[slice_h, slice_w, cur_z_index]
                cur_indices = tf.reshape(cur_indices, (-1, 3))
                c1_region = c1_transposed[slice_h, slice_w, :, :]
                c1_region = tf.reshape(c1_region, (-1, num_channels, batch_size))
                c2_region = c2_transposed[slice_h_r, slice_w_r, :, :]
                c2_region = tf.reshape(c2_region, (-1, num_channels, batch_size))
                all_indices.append(cur_indices)
                all_c1_regions.append(c1_region)
                all_c2_regions.append(c2_region)
                cur_z_index -= 1

        all_indices = tf.concat(all_indices[::-1], axis=0)
        all_c1_regions = tf.concat(all_c1_regions[::-1], axis=0)
        all_c2_regions = tf.concat(all_c2_regions[::-1], axis=0)
        all_costs = tf.reduce_mean(all_c1_regions * all_c2_regions, axis=1)
        batch_dim = tf.shape(c1)[0]
        target_shape = [cv_height, cv_width, square_area, batch_dim]
        cv = tf.scatter_nd(all_indices, all_costs, target_shape)
        cv = tf.transpose(cv, [3, 0, 1, 2])
        return cv
