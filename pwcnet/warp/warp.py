import tensorflow as tf
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network


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
        return spatial_transformer_network(images, optical_flows, True, bilinear_sample=bilinear_sample)
