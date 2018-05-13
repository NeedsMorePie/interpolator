import numpy as np
import tensorflow as tf


def print_tensor_shape(x):
    print(x.get_shape().as_list())

def tf_coin_flip(heads_rate):
    rand_val = tf.random_uniform([1], minval=0.0, maxval=1.0)
    is_head = tf.less(rand_val, heads_rate)
    return tf.cast(is_head, tf.float32)