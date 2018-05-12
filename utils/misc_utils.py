import numpy as np
import tensorflow as tf


def print_tensor_shape(x):
    print(x.get_shape().as_list())