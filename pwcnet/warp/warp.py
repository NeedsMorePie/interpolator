import tensorflow as tf
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network


def optical_flow_to_transforms(optical_flows):
    """
    Outputs the optical flow transform in normalized coordinates.
    :param optical_flows: Tensor of shape (Batch, Height, Width, 2).
    :return: Tensor of shape (Batch, Height, Width, 2). Contents are normalized.
    """
    optical_flow_shape = tf.shape(optical_flows)
    H = optical_flow_shape[1]
    W = optical_flow_shape[2]
    x_trans = optical_flows[..., 0]
    y_trans = optical_flows[..., 1]
    # Note that the * 2.0 is because the normalized coordinates are bound between [-1, 1] and not [0, 1].
    optical_transforms = tf.stack([x_trans / tf.cast(W, dtype=tf.float32) * 2.0,
                                   y_trans / tf.cast(H, dtype=tf.float32) * 2.0], axis=-1)

    return optical_transforms


def optical_flow_to_transforms_immediate(optical_flows, session):
    """
    Creates and runs a small optical_flow_to_transforms graph.
    :param optical_flows: Np array of shape (Batch, Height, Width, 2).
    :param session: Tensorflow session.
    :return: Np array of shape (Batch, Height, Width, 6).
    """
    input_tensor = tf.placeholder(shape=optical_flows.shape, dtype=tf.float32)
    output_tensor = optical_flow_to_transforms(input_tensor)
    return session.run(output_tensor, feed_dict={input_tensor: optical_flows})


def warp_via_flow(images, optical_flows, bilinear_sample=True):
    """
    Given a flow at image A that flows from B to A,
    warp image B to image A.
    :param images: Tensor of shape (Batch, Height, Width, Channels).
    :param optical_flows: Tensor of shape (Batch, Height, Width, 2).
    :param bilinear_sample: Whether to use bilinear interpolation sampling or just nearest-pixel sampling.
    :return: Warped images -- tensors of shape (Batch, Height, Width, Channels).
    """
    with tf.name_scope('warp'):
        transforms = optical_flow_to_transforms(optical_flows)
        return spatial_transformer_network(images, transforms, True, bilinear_sample=bilinear_sample)
