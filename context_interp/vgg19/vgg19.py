# Copied from https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19_trainable.py
# Commit 4006debc1317b2de2b04ffa37c2040f9e2c1e4ad.
# Modifications include adding partial build functions and moving trainable argument to build functions.
import os
import time
import inspect
from utils.misc import print_tensor_shape
import tensorflow as tf
import numpy as np
from functools import reduce

VERBOSE = False
VGG_MEAN = [103.939, 116.779, 123.68]

_default = object()

class Vgg19:
    """
    A trainable version VGG19.
    """
    def __init__(self, vgg19_npy_path=_default, dropout=0.5):
        if vgg19_npy_path is not None:
            if vgg19_npy_path == _default:
                path = inspect.getfile(Vgg19)
                path = os.path.abspath(os.path.join(path, os.pardir))
                path = os.path.join(path, "vgg19.npy")
                vgg19_npy_path = path
            now = time.time()
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

            if VERBOSE:
                print('Loaded from ' + vgg19_npy_path + ' in %d' % time.time() - now)
        else:
            self.data_dict = None

        self.var_dict = {}
        self.dropout = dropout

    def build_up_to_conv1_2(self, rgb, trainable=True):
        """
        Build VGG19 partially, up to the conv1_2 layer.
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1].
        :param trainable: Whether the model is trainable.
        """
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", trainable)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", trainable)

    def build_up_to_conv4_4(self, rgb, trainable=True):
        self.build_up_to_conv1_2(rgb, trainable=trainable)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", trainable)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", trainable)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", trainable)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", trainable)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", trainable)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", trainable)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", trainable)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", trainable)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", trainable)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", trainable)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", trainable)

    def build(self, rgb, trainable=True, train_mode=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param trainable: Whether the model is trainable.
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", trainable)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", trainable)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", trainable)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", trainable)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", trainable)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", trainable)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", trainable)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", trainable)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", trainable)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", trainable)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", trainable)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", trainable)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", trainable)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", trainable)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", trainable)
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", trainable)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2, trainable) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", trainable)
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8", trainable)

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, trainable):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name, trainable):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            assert trainable

        if trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("File saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count