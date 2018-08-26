# Taken from https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
# Commit 8bff4939963c7d0adb9435880dc506fb3f988080
import tensorflow as tf
import numpy as np
from common.forward_warp.forward_warp import create_disocclusion_map
from pwcnet.warp.warp import backward_warp
from tensorflow.contrib.distributions import Normal


def length_sq(x):
    """
    :param x: Tensor.
    :return: Scalar tensor.
    """
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)


def compute_losses(im1, im2, flow_fw, flow_bw,
                   border_mask=None,
                   mask_occlusion='',
                   data_max_distance=1):
    """
    Creates unsupervised UnFlow losses as seen in the UnFlow paper: https://arxiv.org/pdf/1711.07837.pdf.
    :param im1: Tensor of shape [B, H, W, 3].
    :param im2: Tensor of shape [B, H, W, 3].
    :param flow_fw: Tensor of shape [B, H, W, 2].
    :param flow_bw: Tensor of shape [B, H, W, 2].
    :param border_mask: Tensor of shape [B, H, W, 1].
    :param mask_occlusion: Str. Either 'fb', 'disocc', or ''.
    :param data_max_distance: Int.
    :return: Dict. Contains loss terms.
    """
    losses = {}

    im2_warped = backward_warp(im2, flow_fw)
    im1_warped = backward_warp(im1, flow_bw)

    im_diff_fw = im1 - im2_warped
    im_diff_bw = im2 - im1_warped

    disocc_fw = create_disocclusion_map(flow_fw)
    disocc_bw = create_disocclusion_map(flow_bw)

    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask
        mask_bw = border_mask

    fb_occ_fw, fb_occ_bw, flow_diff_fw, flow_diff_bw = occlusion(flow_fw, flow_bw)

    if mask_occlusion == 'fb':
        mask_fw *= (1 - fb_occ_fw)
        mask_bw *= (1 - fb_occ_bw)
    elif mask_occlusion == 'disocc':
        mask_fw *= (1 - disocc_bw)
        mask_bw *= (1 - disocc_fw)

    occ_fw = 1 - mask_fw
    occ_bw = 1 - mask_bw

    losses['sym'] = (charbonnier_loss(occ_fw - disocc_bw) +
                     charbonnier_loss(occ_bw - disocc_fw))

    losses['occ'] = (charbonnier_loss(occ_fw) +
                     charbonnier_loss(occ_bw))

    losses['photo'] = (photometric_loss(im_diff_fw, mask_fw) +
                       photometric_loss(im_diff_bw, mask_bw))

    losses['grad'] = (gradient_loss(im1, im2_warped, mask_fw) +
                      gradient_loss(im2, im1_warped, mask_bw))

    losses['smooth_1st'] = (smoothness_loss(flow_fw) +
                            smoothness_loss(flow_bw))

    losses['smooth_2nd'] = (second_order_loss(flow_fw) +
                            second_order_loss(flow_bw))

    losses['fb'] = (charbonnier_loss(flow_diff_fw, mask_fw) +
                    charbonnier_loss(flow_diff_bw, mask_bw))

    losses['ternary'] = (ternary_loss(im1, im2_warped, mask_fw,
                                      max_distance=data_max_distance) +
                         ternary_loss(im2, im1_warped, mask_bw,
                                      max_distance=data_max_distance))

    return losses


def ternary_loss(im1, im2_warped, mask, max_distance=1):
    """
    :param im1: Tensor of shape [B, H, W, 3].
    :param im2_warped: Tensor of shape [B, H, W, 3].
    :param mask: Tensor of shape [B, H, W, 1].
    :param max_distance: Int.
    :return: Scalar tensor.
    """
    patch_size = 2 * max_distance + 1
    with tf.variable_scope('ternary_loss'):
        def _ternary_transform(image):
            intensities = tf.image.rgb_to_grayscale(image) * 255
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
            weights = tf.constant(w, dtype=tf.float32)
            patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

            transf = patches - intensities
            transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = tf.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
            return dist_sum

        t1 = _ternary_transform(im1)
        t2 = _ternary_transform(im2_warped)
        dist = _hamming_distance(t1, t2)

        transform_mask = create_mask(mask, [[max_distance, max_distance],
                                            [max_distance, max_distance]])
        return charbonnier_loss(dist, mask * transform_mask)


def occlusion(flow_fw, flow_bw):
    """
    :param flow_fw: Tensor of shape [B, H, W, 2].
    :param flow_bw: Tensor of shape [B, H, W, 2].
    :return: occ_fw: Tensor of shape [B, H, W, 1].
             occ_bw: Tensor of shape [B, H, W, 1].
             flow_diff_fw: Tensor of shape [B, H, W, 2]. Difference between backward and forward flows.
             flow_diff_bw. Tensor of shape [B, H, W, 2]. Difference between backward and forward flows.
    """
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = backward_warp(flow_bw, flow_fw)
    flow_fw_warped = backward_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh = 0.01 * mag_sq + 0.5
    occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh, tf.float32)
    occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh, tf.float32)
    return occ_fw, occ_bw, flow_diff_fw, flow_diff_bw


def norm(x, sigma):
    """
    Gaussian decay.
    Result is 1.0 for x = 0 and decays towards 0 for |x > sigma.
    :param x: Tensor.
    :param sigma: Tensor.
    :return: Tensor.
    """
    dist = Normal(0.0, sigma)
    return dist.pdf(x) / dist.pdf(0.0)


def diffusion_loss(flow, im, occ):
    """
    Forces diffusion weighted by motion, intensity and occlusion label similarity.
    Inspired by Bilateral Flow Filtering.
    :param flow: Tensor of shape [B, H, W, 2].
    :param im: Tensor of shape [B, H, W, 3].
    :param occ: Tensor of shape [B, H, W, 1].
    :return: Scalar tensor.
    """
    def neighbor_diff(x, num_in=1):
        weights = np.zeros([3, 3, num_in, 8 * num_in])
        out_channel = 0
        for c in range(num_in): # over input channels
            for n in [0, 1, 2, 3, 5, 6, 7, 8]: # over neighbors
                weights[1, 1, c, out_channel] = 1
                weights[n // 3, n % 3, c, out_channel] = -1
                out_channel += 1
        weights = tf.constant(weights, dtype=tf.float32)
        return conv2d(x, weights)

    # Create 8 channel (one per neighbor) differences
    occ_diff = neighbor_diff(occ)
    flow_diff_u, flow_diff_v = tf.split(axis=3, num_or_size_splits=2, value=neighbor_diff(flow, 2))
    flow_diff = tf.sqrt(tf.square(flow_diff_u) + tf.square(flow_diff_v))
    intensity_diff = tf.abs(neighbor_diff(tf.image.rgb_to_grayscale(im)))

    diff = norm(intensity_diff, 7.5 / 255) * norm(flow_diff, 0.5) * occ_diff * flow_diff
    return charbonnier_loss(diff)


def photometric_loss(im_diff, mask):
    """
    :param im_diff: Tensor of shape [B, H, W, 3].
    :param mask: Tensor of shape [B, H, W, 1].
    :return: Scalar tensor.
    """
    return charbonnier_loss(im_diff, mask, beta=255)


def conv2d(x, weights):
    """
    :param x: Tensor of shape [B, H, W, C_input].
    :param weights: Tensor of shape [filter_height, filter_width, C_input, C_output]
    :return: Tensor of shape [B, H, W, C_output].
    """
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')


def _smoothness_deltas(flow):
    """
    :param flow: Tensor of shape [B, H, W, 2].
    :return: delta_u: Tensor of shape [B, H, W, 2].
             delta_v: Tensor of shape [B, H, W, 2].
             mask: Tensor.
    """
    with tf.variable_scope('smoothness_delta'):
        mask_x = create_mask(flow, [[0, 0], [0, 1]])
        mask_y = create_mask(flow, [[0, 1], [0, 0]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y])

        filter_x = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        filter_y = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)
        return delta_u, delta_v, mask


def _gradient_delta(im1, im2_warped):
    """
    :param im1: Tensor of shape [B, H, W, 3].
    :param im2_warped: Tensor of shape [B, H, W, 3].
    :return: Tensor of shape [B, H, W, 6].
    """
    with tf.variable_scope('gradient_delta'):
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array = np.zeros([3, 3, 3, 6])
        for c in range(3):
            weight_array[:, :, c, 2 * c] = filter_x
            weight_array[:, :, c, 2 * c + 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        im1_grad = conv2d(im1, weights)
        im2_warped_grad = conv2d(im2_warped, weights)
        diff = im1_grad - im2_warped_grad
        return diff


def gradient_loss(im1, im2_warped, mask):
    """
    :param im1: Tensor of shape [B, H, W, 3].
    :param im2_warped: Tensor of shape [B, H, W, 3].
    :param mask: Tensor.
    :return: Scalar tensor.
    """
    with tf.variable_scope('gradient_loss'):
        mask_x = create_mask(im1, [[0, 0], [1, 1]])
        mask_y = create_mask(im1, [[1, 1], [0, 0]])
        gradient_mask = tf.tile(tf.concat(axis=3, values=[mask_x, mask_y]), [1, 1, 1, 3])
        diff = _gradient_delta(im1, im2_warped)
        return charbonnier_loss(diff, mask * gradient_mask)


def smoothness_loss(flow):
    """
    :param flow: Tensor of shape [B, H, W, 2].
    :return: Scalar tensor.
    """
    with tf.variable_scope('smoothness_loss'):
        delta_u, delta_v, mask = _smoothness_deltas(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        return loss_u + loss_v


def _second_order_deltas(flow):
    """
    :param flow: Tensor of shape [B, H, W, 2].
    :return: delta_u: Tensor of shape [B, H, W, 4].
             delta_v: Tensor of shape [B, H, W, 4].
             mask: Tensor of shape [B, H, W, 4].
    """
    with tf.variable_scope('_second_order_deltas'):
        mask_x = create_mask(flow, [[0, 0], [1, 1]])
        mask_y = create_mask(flow, [[1, 1], [0, 0]])
        mask_diag = create_mask(flow, [[1, 1], [1, 1]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y, mask_diag, mask_diag])

        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)
        return delta_u, delta_v, mask


def second_order_loss(flow):
    """
    :param flow: Tensor of shape [B, H, W, 2].
    :return: Scalar tensor.
    """
    with tf.variable_scope('second_order_loss'):
        delta_u, delta_v, mask = _second_order_deltas(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    :param x: Tensor of shape [B, H, W, C].
    :param mask: Tensor of shape [B, H, W, C].
    :param truncate: Scalar tensor or None.
    :param alpha: Scalar tensor.
    :param beta: Scalar tensor.
    :param epsilon: Scalar tensor.
    :return: Scalar tensor.
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return tf.reduce_sum(error) / normalization


def create_mask(tensor, paddings):
    """
    :param tensor: Tensor.
    :param paddings: 2D array of floats.
    :return: Tensor.
    """
    with tf.variable_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)


def create_border_mask(tensor, border_ratio=0.1):
    """
    :param tensor: Tensor of shape [B, H, W, C].
    :param border_ratio: Float or scalar tensor.
    :return: Tensor.
    """
    with tf.variable_scope('create_border_mask'):
        num_batch, height, width, _ = tf.unstack(tf.shape(tensor))
        min_dim = tf.cast(tf.minimum(height, width), 'float32')
        sz = tf.cast(tf.ceil(min_dim * border_ratio), 'int32')
        border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
        return tf.stop_gradient(border_mask)


def create_outgoing_mask(flow):
    """
    Computes a mask that is zero at all positions where the flow would carry a pixel over the image boundary.
    :param flow: Tensor of shape [B, H, W, 2].
    :return: Tensor.
    """
    with tf.variable_scope('create_outgoing_mask'):
        num_batch, height, width, _ = tf.unstack(tf.shape(flow))

        grid_x = tf.reshape(tf.range(width), [1, 1, width])
        grid_x = tf.tile(grid_x, [num_batch, height, 1])
        grid_y = tf.reshape(tf.range(height), [1, height, 1])
        grid_y = tf.tile(grid_y, [num_batch, 1, width])

        flow_u, flow_v = tf.unstack(flow, 2, 3)
        pos_x = tf.cast(grid_x, dtype=tf.float32) + flow_u
        pos_y = tf.cast(grid_y, dtype=tf.float32) + flow_v
        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >= 0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >= 0.0)
        inside = tf.logical_and(inside_x, inside_y)
        return tf.expand_dims(tf.cast(inside, tf.float32), 3)