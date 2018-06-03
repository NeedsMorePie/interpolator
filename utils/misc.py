import numpy as np
import tensorflow as tf


# https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w
def sort_in_unison(key_list, lists):
    """
    :param key_list: The list whose elements we use to sort.
    :param lists: A list of lists, each of which will be sorted in unison with key_list.
    :return: (sorted_key_list, sorted_lists)
    """
    indexes = [*range(len(key_list))]
    indexes.sort(key=key_list.__getitem__)
    sorted_lists = []
    sorted_key_list = [key_list[indexes[i]] for i in range(len(key_list))]
    for list in lists:
        sorted_list = [list[indexes[i]] for i in range(len(list))]
        sorted_lists.append(sorted_list)
    return sorted_key_list, sorted_lists


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
