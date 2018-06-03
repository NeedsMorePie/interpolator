import unittest
import os
from utils.flow import read_flow_file
from utils.img import read_image, show_image
from utils.misc import sort_in_unison
from experiments.forward_warp_np import forward_warp_np
from experiments.forward_warp import forward_warp, get_pushed_pixels
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import cv2

VISUALIZE = False


class TestForwardWarp(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_push_pixels_whole_1(self):
        height = 2
        width = 2
        features = [
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]]
        ]

        # Flow is in (x, y) order.
        # Flows first column to second column.
        flow = [
            [[1, 0], [0, 0]],
            [[1, 0], [0, 0]]
        ]

        # Indices are in (y, x) order.
        # As the pixels are splatted on at exact integer coordinates, there are only 4 resulting targets.
        expected_indices = [
            [0, 1], [0, 1], [1, 1], [1, 1]
        ]

        expected_values = [
            [1, 0], [0, 0], [0, 1], [0, 0]
        ]

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        flow_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_pushed_pixels(features_tensor, flow_tensor, 1.0)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features, flow_tensor: flow})

        flattened_indices = [index[0] * width + index[1] for index in indices]
        _, sorted = sort_in_unison(flattened_indices, [indices, values])
        indices, values = sorted

        self.assertListEqual(indices.tolist(), expected_indices)
        self.assertListEqual(values.tolist(), expected_values)

    def test_push_pixels_whole_2(self):
        height = 2
        width = 3
        features = [
            [[1, 0], [0, 0], [-1, 0]],
            [[0, 1], [0, 0], [-1, 0]]
        ]

        # Flow is in (x, y) order.
        # Flows first column to second column.
        flow = [
            [[1, 0], [0, 0], [0, 1]],
            [[1, 0], [0, 0], [0, 0]]
        ]

        # Indices are in (y, x) order.
        # As the pixels are splatted on at exact integer coordinates, there are only 4 resulting targets.
        expected_indices = [
            [0, 1], [0, 1], [1, 2], [1, 1], [1, 1], [1, 2]
        ]

        expected_values = [
            [1, 0], [0, 0], [-1, 0], [0, 1], [0, 0], [-1, 0]
        ]

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        flow_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_pushed_pixels(features_tensor, flow_tensor, 1.0)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features, flow_tensor: flow})

        flattened_indices = [index[0] * width + index[1] for index in indices]
        _, sorted = sort_in_unison(flattened_indices, [indices, values])
        indices, values = sorted

        self.assertListEqual(indices.tolist(), expected_indices)
        self.assertListEqual(values.tolist(), expected_values)


    def test_visualization(self):
        if not VISUALIZE:
            return

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(cur_dir, '..')
        flow_ab = read_flow_file(root_dir + 'pwcnet/warp/test_data/flow_ab.flo')
        img_a = read_image(root_dir + 'pwcnet/warp/test_data/image_a.png', as_float=True)
        warped = forward_warp(img_a, flow_ab, 1.0)
        warped = np.clip(warped, 0.0, 1.0)
        show_image(warped)

        # For writing to video.
        # height = img_a.shape[0]
        # width = img_a.shape[1]
        # writer = cv2.VideoWriter(cur_dir + '/outputs/video.avi', cv2.VideoWriter_fourcc(*"MJPG"), 20, (width, height))
        #
        # steps = 20
        # for i in range(steps):
        #     t = i * (1.0 / float(steps))
        #     warped = forward_warp(img_a, flow_ab, t)
        #     warped = np.clip(warped, 0.0, 1.0)
        #     output_path = cur_dir + "/outputs/out-%.2f.png" % t
        #     mpimg.imsave(output_path, warped)
        #     writer.write(cv2.imread(output_path))
        #
        # writer.release()
