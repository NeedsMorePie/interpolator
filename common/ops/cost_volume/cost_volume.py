# Mostly copied from https://github.com/nameless-Chatoyant/PWC-Net_pytorch/blob/master/modules.py.
# Master branch commit 2225ad2082371126cc9c8e57a8b962a88933a8c0.
import tensorflow as tf
import os
from tensorflow.python.framework import ops
from utils.misc import *


def cost_volume(c1, c2, search_range=4):
    """
    See https://arxiv.org/pdf/1709.02371.pdf.
    For each pixel in c1, we will compute correlations with its spatial neighbors in c2.
    :param c1: Tensor. Feature map of shape [batch_size, H, W, num_features].
    :param c2: Input tensor with the exact same shape as c1.
    :param search_range: The search square's side length is equal to 2 * search_range + 1.
    :return: Tensor. Cost volume of shape [batch_size, H, W, s * s], where s is equal to 2 * search_range + 1.
    """
    c1 = tf.transpose(c1, [0, 3, 1, 2])
    c2 = tf.transpose(c2, [0, 3, 1, 2])
    results = mod.correlation(
        c1, c2,
        max_displacement=search_range,
        pad=search_range,
        stride_1=1,
        stride_2=1
    )
    return tf.transpose(results[0], [0, 2, 3, 1])
    

# Load op and register gradients.
mod = tf.load_op_library(os.path.join('build', 'libcorrelation_op.so'))


@ops.RegisterGradient("Correlation")
def _CorrelationGrad(op, in_grad, in_grad1, in_grad2):
    grad0, grad1 = mod.correlation_grad(
        in_grad, op.inputs[0], op.inputs[1],
        op.outputs[1], op.outputs[2],
        kernel_size=op.get_attr('kernel_size'),
        max_displacement=op.get_attr('max_displacement'),
        pad=op.get_attr('pad'),
        stride_1=op.get_attr('stride_1'),
        stride_2=op.get_attr('stride_2'))
    return [grad0, grad1]
