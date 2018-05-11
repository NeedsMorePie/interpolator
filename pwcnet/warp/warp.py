import tensorflow as tf
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network


def optical_flow_to_transforms(optical_flows):
    """
    Converts the optical flow image to a transform matrix.
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
    optical_transforms = tf.stack([ones, zeros, -x_trans / tf.cast(W, dtype=tf.float32) * 2.0,
                                   zeros, ones, -y_trans / tf.cast(H, dtype=tf.float32) * 2.0], axis=-1)

    return optical_transforms


def warp_via_flow(images, optical_flows):
    """
    :param images: Tensor of shape (Batch, Height, Width, Channels).
    :param optical_flows: Tensor of shape (Batch, Height, Width, 2).
    :return: Warped images -- tensors of shape (Batch, Height, Width, Channels).
    """
    transforms = optical_flow_to_transforms(optical_flows)
    return spatial_transformer_network(images, transforms, True)
