import tensorflow as tf
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network


def optical_flow_to_transforms(optical_flows):
    """
    Converts the optical flow image to a flattened 2D transform matrix.
    Outputs the transform in normalized coordinates.
    :param optical_flows: Tensor of shape (Batch, Height, Width, 2).
    :return: Tensor of shape (Batch, Height, Width, 6).
    """
    optical_flow_shape = tf.shape(optical_flows)
    B = optical_flow_shape[0]
    H = optical_flow_shape[1]
    W = optical_flow_shape[2]
    zeros = tf.zeros(shape=(B, H, W))
    ones = tf.ones(shape=(B, H, W))
    x_trans = optical_flows[..., 0]
    y_trans = optical_flows[..., 1]
    # Note that the * 2.0 is because the normalized coordinates are bound between [-1, 1] and not [0, 1].
    optical_transforms = tf.stack([ones, zeros, -x_trans / tf.cast(W, dtype=tf.float32) * 2.0,
                                   zeros, ones, -y_trans / tf.cast(H, dtype=tf.float32) * 2.0], axis=-1)

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


def warp_via_flow(images, optical_flows):
    """
    :param images: Tensor of shape (Batch, Height, Width, Channels).
    :param optical_flows: Tensor of shape (Batch, Height, Width, 2).
    :return: Warped images -- tensors of shape (Batch, Height, Width, Channels).
    """
    transforms = optical_flow_to_transforms(optical_flows)
    return spatial_transformer_network(images, transforms, True)
