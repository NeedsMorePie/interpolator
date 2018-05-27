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
def prelu(_x):
    with tf.variable_scope(_x.op.name + '_activation', initializer=tf.constant_initializer(1.0)):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg


def sliding_window_slice(x, slice_locations):
    """
    :param x: The tensor to window slice.
    :param slice_locations: A list. The locations at which we gather values. Values are either 0 or 1.
                            E.g [1, 0, 1] means that at each window offset j, we form [x[j], x[j + 2]].
    :return: The window sliced tensor. Is 1 rank higher than x.
    """
    slice_indices = []
    for i in range(len(slice_locations)):
        if slice_locations[i] == 1:
            slice_indices.append(i)

    sequence_len = len(slice_indices)

    # Get the sliding window indices.
    slice_indices_tensor = tf.constant(slice_indices)
    num_offsets = tf.shape(x)[0] - tf.cast(len(slice_locations) - 1, tf.int32)

    def get_zeros(sequence_len, x):
        return tf.zeros(tf.concat([[1, sequence_len], tf.shape(x)[1:]], axis=0))

    def get_slice(slice_indices_tensor, num_offsets, x):
        tiled = tf.tile(slice_indices_tensor, [num_offsets])
        tiled = tf.expand_dims(tiled, axis=0)

        tiled = tf.reshape(tiled, [num_offsets, -1])
        offsets = tf.expand_dims(tf.range(0, num_offsets), axis=1)
        indices = tiled + offsets
        indices = tf.reshape(indices, [-1])

        # Gather and reshape.
        images = tf.gather(x, indices)
        images = tf.expand_dims(images, axis=0)
        final_shape = tf.concat([[num_offsets, sequence_len], tf.shape(images)[2:]], axis=0)
        images = tf.reshape(images, final_shape)
        return tf.cast(images, tf.float32)

    slices = tf.cond(
        num_offsets > 0,
        true_fn=lambda: get_slice(slice_indices_tensor, num_offsets, x),
        false_fn=lambda: get_zeros(sequence_len, x)
    )
    return slices