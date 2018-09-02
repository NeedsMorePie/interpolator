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


def create_multi_level_loss(expected_flow, flows, flow_scaling, flow_layer_loss_weights=None, diff_fn=l2_diff):
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
    if flow_layer_loss_weights is None:
        flow_layer_loss_weights = [0.32, 0.08, 0.02, 0.01, 0.0025, 0.005]

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
    :param forward_flows: List of tensors of shape [B, H_i, W_i, 2]. Flows at different resolutions (i.e. different
                          H_i, W_i for different resolutions). By default the resolutions should be increasing.
                          These should be in in the same order as flow_layer_loss_weights.
    :param backward_flows: List of tensors of shape [B, H_i, W_i, 2]. Flows at different resolutions (i.e. different
                           H_i, W_i for different resolutions). By default the resolutions should be increasing.
                           These should be in in the same order as flow_layer_loss_weights.
    :param flow_scaling: Float. Magnitude scale of the forward/backward flows with respect to the final flow. The
                         forward/backward flows will be divided by this before they are used in this function.
    :param flow_layer_loss_weights: List of floats. Loss weights that correspond to the flows list.
    :param layer_patch_distances: List of ints. These should be in the same order as flow_layer_loss_weights.
    :param loss_weights: Dict. Key is the loss name and the value is the weight.
    :return: total_loss: Scalar tensor. Sum off all weighted level losses.
             layer_losses: List of scalar tensors. Weighted loss at each level.
             forward_occlusion_masks: List of tensors of shape [B, H, W, 1].
             backward_occlusion_masks: List of tensors of shape [B, H, W, 1].
             layer_losses_detailed: List of dicts. Each dict contains the individual fully weighted losses.
    """
    # Set defaults.
    if flow_layer_loss_weights is None:
        flow_layer_loss_weights = [1.1, 3.4, 3.9, 4.35, 4.35, 12.7]
    if layer_patch_distances is None:
        layer_patch_distances = [1, 1, 2, 2, 3, 3]
    if loss_weights is None:
        loss_weights = {
            'ternary': 1.0,  # E_D data loss term in the paper.
            'occ': 12.4,  # Lambda_p. This is part of the E_D term to penalize the trivial occlusion mask.
            'smooth_2nd': 3.0,  # E_S second order smoothness constraint in the paper.
            'fb': 0.2  # E_C forward-backward consistency term in the paper.
        }

    # Length assertions.
    assert len(flow_layer_loss_weights) == len(layer_patch_distances)
    assert len(forward_flows) == len(layer_patch_distances)
    assert len(forward_flows) == len(backward_flows)

    layer_losses = []
    forward_occlusion_masks = []
    backward_occlusion_masks = []
    layer_losses_detailed = []

    _, image_height, _, _ = tf.unstack(tf.shape(image_a))
    for i, (forward_flow, backward_flow) in enumerate(zip(forward_flows, backward_flows)):
        if flow_layer_loss_weights[i] == 0.0:
            continue
        with tf.name_scope('layer_' + str(i) + '_loss'):
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
            losses, occ_fw, occ_bw = compute_losses(resize_image_a, resize_image_b, unscaled_forward_flow,
                                                    unscaled_backward_flow, prewarp_scaling=res_scaling,
                                                    data_max_distance=layer_patch_distances[i])
            forward_occlusion_masks.append(occ_fw)
            backward_occlusion_masks.append(occ_bw)

            # Get losses for this layer.
            this_layer_losses = []
            this_layer_losses_dict = {}
            for key in loss_weights.keys():
                assert key in losses
                loss = loss_weights[key] * losses[key]
                this_layer_losses.append(loss)
                this_layer_losses_dict[key] = flow_layer_loss_weights[i] * loss
            if len(this_layer_losses) > 0:
                # Sum all losses for this layer and apply the loss weight.
                layer_losses.append(flow_layer_loss_weights[i] * tf.add_n(this_layer_losses))
                layer_losses_detailed.append(this_layer_losses_dict)

    # Sum up the total loss.
    if len(layer_losses) > 0:
        total_loss = tf.add_n(layer_losses, name='total_unflow_loss')
    else:
        total_loss = tf.constant(0.0, dtype=tf.float32)

    return total_loss, layer_losses, forward_occlusion_masks, backward_occlusion_masks, layer_losses_detailed
