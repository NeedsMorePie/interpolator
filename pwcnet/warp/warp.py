import tensorflow as tf
import os.path
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network
from sys import platform
from tensorflow.python.framework import ops


# Load op library.
if platform == 'win32':
    lib_path = os.path.join('build', 'backward_warp_op.dll')
else:
    lib_path = os.path.join('build', 'libbackward_warp_op.so')
if os.path.isfile(lib_path):
    mod = tf.load_op_library(lib_path)
else:
    print('Warning: No CUDA implementation of backward_warp found. Falling back to the Tensorflow version.')
    mod = None


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
