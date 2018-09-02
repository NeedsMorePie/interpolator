import tensorflow as tf
from common.utils.tf import load_op_library
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network
from tensorflow.python.framework import ops


# Load op library.
mod = load_op_library('backward_warp_op', 'build')


def backward_warp(images, optical_flows, bilinear_sample=True):
    """
    Given a flow at image A that flows from B to A,
    warp image B to image A.
    :param images: Tensor of shape (Batch, Height, Width, Channels).
    :param optical_flows: Tensor of shape (Batch, Height, Width, 2).
    :param bilinear_sample: Whether to use bilinear interpolation sampling or just nearest-pixel sampling.
    :return: Warped images -- tensors of shape (Batch, Height, Width, Channels).
    """
    if mod is not None:
        return mod.backward_warp(images, optical_flows)
    else:
        with tf.name_scope('warp'):
            return spatial_transformer_network(images, optical_flows, True, bilinear_sample=bilinear_sample)


if mod is not None:
    @ops.RegisterGradient('BackwardWarp')
    def _BackwardWarpGrad(op, grad):
        image_grad, flow_grad = mod.backward_warp_grad(
            grad, op.inputs[0], op.inputs[1])
        return [image_grad, flow_grad]
