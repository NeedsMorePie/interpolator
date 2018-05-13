import numpy as np
import tensorflow as tf


def print_tensor_shape(x):
    print(x.get_shape().as_list())

def tf_coin_flip(heads_rate):
    rand_val = tf.random_uniform([1], minval=0.0, maxval=1.0)
    is_head = tf.less(rand_val, heads_rate)
    return is_head

# https://github.com/tensorflow/tensorflow/issues/7712
def pelu(x):
    """Parametric Exponential Linear Unit (https://arxiv.org/abs/1605.09332v1)."""
    with tf.variable_scope(x.op.name + '_activation', initializer=tf.constant_initializer(1.0)):
        shape = x.get_shape().as_list()[1:]
        alpha = tf.get_variable('alpha', shape)
        beta = tf.get_variable('beta', shape)
        positive = tf.nn.relu(x) * alpha / (beta + 1e-9)
        negative = alpha * (tf.exp((-tf.nn.relu(-x)) / (beta + 1e-9)) - 1)
        return negative + positive

# https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
def parametric_relu(_x):
    with tf.variable_scope(_x.op.name + '_activation', initializer=tf.constant_initializer(1.0)):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg