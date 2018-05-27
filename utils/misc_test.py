import unittest
import tensorflow as tf
import numpy as np
from utils.misc import *

class TestMiscUtils(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_sliding_window_slice_dense(self):
        x = tf.constant([
            [0, 0],
            [0, 1],
            [1, 0],
        ])
        expected = [
            [[0, 0], [0, 1]],
            [[0, 1], [1, 0]]
        ]
        slice_locations = [1, 1]
        sliced = sliding_window_slice(x, slice_locations)
        sliced_list = self.sess.run(sliced).tolist()
        self.assertListEqual(sliced_list, expected)

    def test_sliding_window_slice_sparse(self):
        x = tf.constant([
            [0, 0],
            [1, 1],
            [0, 0],
            [2, 2],
            [0, 0],
            [3, 3],
        ])
        expected = [
            [[0, 0], [0, 0], [0, 0]],
            [[1, 1], [2, 2], [3, 3]]
        ]
        slice_locations = [1, 0, 1, 0, 1]
        sliced = sliding_window_slice(x, slice_locations)
        sliced_list = self.sess.run(sliced).tolist()
        self.assertListEqual(sliced_list, expected)

    def test_sliding_window_slice_small(self):
        x = tf.constant([
            [1, 2.4]
        ])
        expected = [
            [[0, 0], [0, 0], [0, 0]]
        ]
        slice_locations = [1, 1, 1]
        sliced = sliding_window_slice(x, slice_locations)
        sliced_list = self.sess.run(sliced).tolist()
        self.assertListEqual(sliced_list, expected)
