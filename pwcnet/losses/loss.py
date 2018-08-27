import tensorflow as tf
from pwcnet.losses.unflow import compute_losses


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
    :param flow_scaling: Float. Scales the expected_flow before comparing it to the flows.
    :param flow_layer_loss_weights: List of floats. Loss weights that correspond to the flows list.
    :param diff_fn: Function.
    :return: total_loss: Scalar tensor. Sum off all weighted level losses.
             layer_losses: List of scalar tensors. Weighted loss at each level.
    """
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
            layer_losses.append(layer_loss)

    total_loss = tf.add_n(layer_losses, name='total_loss')
    return total_loss, layer_losses


def create_multi_level_unflow_loss(image_a, image_b, forward_flows, backward_flows, flow_scaling,
                                   flow_layer_loss_weights=None, layer_patch_distances=None, loss_weights=None):
    """
    :param image_a: Tensor of shape [B, H, W, 3].
    :param image_b: Tensor of shape [B, H, W, 3].
    :param forward_flows: List of tensors of shape [B, H, W, 2]. Flows at different resolutions.
    :param backward_flows: List of tensors of shape [B, H, W, 2]. Flows at different resolutions.
    :param flow_scaling: Float. Scales the expected_flow before comparing it to the flows.
    :param flow_layer_loss_weights: List of floats. Loss weights that correspond to the flows list.
    :param layer_patch_distances: List of ints.
    :param loss_weights: Dict. Key is the loss name and the value is the weight.
    :return: total_loss: Scalar tensor. Sum off all weighted level losses.
             layer_losses: List of scalar tensors. Weighted loss at each level.
    """
    # Set defaults.
    if flow_layer_loss_weights is None:
        flow_layer_loss_weights = [1.1, 3.4, 3.9, 4.35, 12.7]
    if layer_patch_distances is None:
        layer_patch_distances = [1, 1, 2, 2, 3]
    if loss_weights is None:
        loss_weights = {
            'ternary': 1.0,  # E_D data loss term in the paper.
            'occ': 12.4,  # Part of the E_D term to penalize the trivial occlusion mask.
            'smooth_2nd': 3.0,  # E_S second order smoothness constraint in the paper.
            'fb': 0.2  # E_C forward-backward consistency term in the paper.
        }

    # Length assertions.
    assert len(flow_layer_loss_weights) == len(layer_patch_distances)
    assert len(forward_flows) == len(layer_patch_distances)
    assert len(forward_flows) == len(backward_flows)

    layer_losses = []

    _, image_height, _, _ = tf.unstack(tf.shape(image_a))
    for i, (forward_flow, backward_flow) in enumerate(zip(forward_flows, backward_flows)):
        _, flow_height, flow_width, _ = tf.unstack(tf.shape(forward_flow))

        # Resize the input images to match the corresponding flow size.
        resize_image_a = tf.image.resize_bilinear(image_a, (flow_height, flow_width))
        resize_image_b = tf.image.resize_bilinear(image_b, (flow_height, flow_width))

        # We must do this to interop with the rest of the network that uses a scaled flow due to the regular PWC Net
        # loss function.
        unscaled_forward_flow = forward_flow / flow_scaling
        unscaled_backward_flow = backward_flow / flow_scaling

        # The res_scaling is only used directly before using a flow to warp an image.
        # The reason we don't just apply the flow scaling here is because part of the UnFlow loss diffs the flow
        # magnitudes, and it's best to keep the loss magnitudes the same at all levels.
        res_scaling = tf.cast(flow_height, dtype=tf.float32) / tf.cast(image_height, dtype=tf.float32)
        losses = compute_losses(resize_image_a, resize_image_b, unscaled_forward_flow, unscaled_backward_flow,
                                prewarp_scaling=res_scaling, data_max_distance=layer_patch_distances[i])

        # Get losses for this layer.
        this_layer_losses = []
        for key in loss_weights.keys():
            assert key in losses
            this_layer_losses.append(loss_weights[key] * losses[key])
        if len(this_layer_losses) > 0:
            # Sum all losses for this layer and apply the loss weight.
            layer_losses.append(flow_layer_loss_weights[i] * tf.add_n(this_layer_losses))

    # Sum up the total loss.
    if len(layer_losses) > 0:
        total_loss = tf.add_n(layer_losses, name='total_unflow_loss')
    else:
        total_loss = tf.constant(0.0, dtype=tf.float32)

    return total_loss, layer_losses
