import tensorflow as tf


def l2_diff(a, b):
    """
    :param a: Tensor.
    :param b: Tensor.
    :return: Tensor.
    """
    return tf.square(a - b)


def lq_diff(a, b, q=0.4, epsilon=0.01):
    """
    :param a: Tensor
    :param b: Tensor
    :param q: Number or scalar tensor.
    :param epsilon: Number or scalar tensor.
    :return: Tensor.
    """
    return tf.pow(tf.abs(a - b) + epsilon, q)


def create_multi_level_loss(expected_flow, flows, flow_scaling, flow_layer_loss_weights, diff_fn=l2_diff):
    """
    Creates the PWC Net multi-level loss.
    :param expected_flow: Tensor of shape [B, H, W, 2]. Ground truth.
    :param flows: List of flow tensors. Flows at different resolutions.
    :param flow_layer_loss_weights: List of numbers. Loss weights that correspond to the flows list.
    :param flow_scaling: Number. Scales the expected_flow before comparing it to the flows.
    :param diff_fn: Function.
    :return: total_loss: Scalar tensor. Sum off all weighted level losses.
             layer_losses: List of scalar tensors. Weighted loss at each level.
    """
    total_loss = tf.constant(0.0, dtype=tf.float32)
    scaled_gt = expected_flow * flow_scaling

    layer_losses = []
    for i, flow in enumerate(flows):
        if flow_layer_loss_weights[i] == 0:
            continue
        with tf.name_scope('layer_' + str(i) + '_loss'):
            H, W = tf.shape(flow)[1], tf.shape(flow)[2]

            # Ground truth needs to be resized to match the size of the previous flow.
            resized_scaled_gt = tf.image.resize_bilinear(scaled_gt, [H, W])

            # squared_difference has the shape [batch_size, H, W, 2].
            squared_difference = diff_fn(resized_scaled_gt, flow)
            # Reduce sum in the last 3 dimensions, average over the batch, and apply the weight.
            weight = flow_layer_loss_weights[i]
            layer_loss = weight * tf.reduce_mean(tf.reduce_sum(squared_difference, axis=[1, 2, 3]))

            # Accumulate the total loss.
            layer_losses.append(layer_loss)
        total_loss += layer_loss

    return total_loss, layer_losses
