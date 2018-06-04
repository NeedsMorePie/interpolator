import unittest
import os
from utils.flow import read_flow_file
from utils.img import read_image, show_image
from utils.misc import sort_in_unison
from experiments.forward_warp_np import forward_warp_np
from experiments.forward_warp import forward_warp, get_translated_pixels
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
        features = [[
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]]
        ]]

        # Translation is in (y, x) order.
        # Translates first column to second column.
        translations = [[
            [[0, 1], [0, 0]],
            [[0, 1], [0, 0]]
        ]]

        # Indices are in (y, x) order.
        # As the pixels are splatted on at exact integer coordinates, there will be duplicates.
        expected_indices = [[
            [0, 1], [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [0, 1],
            [1, 1], [1, 1], [1, 1], [1, 1],
            [1, 1], [1, 1], [1, 1], [1, 1],
        ]]

        # We expect the duplicates to not splat anything.
        expected_values = [[
            [1, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 1], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-1).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features, 
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-1).tolist()
        self.assertCountEqual(expected, predicted)

    def test_push_pixels_whole_2(self):
        height = 2
        width = 3
        features = [[
            [[1, 0], [0, 0], [-1, 0]],
            [[0, 1], [0, 0], [-1, 0]]
        ]]

        # Translation is in (y, x) order.
        # Translates first column to second column.
        translations = [[
            [[0, 1], [0, 0], [1, 0]],
            [[0, 1], [0, 0], [0, 0]]
        ]]

        # Indices are in (y, x) order.
        # As the pixels are splatted on at exact integer coordinates, there will be duplicates.
        expected_indices = [[
            [0, 1], [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [0, 1],
            [1, 2], [1, 2], [1, 2], [1, 2],
            [1, 1], [1, 1], [1, 1], [1, 1],
            [1, 1], [1, 1], [1, 1], [1, 1],
            [1, 2], [1, 2], [1, 2], [1, 2]
        ]]

        # We expect the duplicates to not splat anything.
        expected_values = [[
            [1, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [-1, 0], [0, 0], [0, 0], [0, 0],
            [0, 1], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [-1, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-1).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features,
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-1).tolist()
        self.assertCountEqual(expected, predicted)

    def test_push_pixels_partial_1(self):
        """
        X or y coordinate are integers, but only 1 of them at a time.
        """
        height = 2
        width = 2
        features = [[
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]]
        ]]

        # Translation is in (y, x) order.
        translations = [[
            [[0.5, 0], [0, 0]],
            [[0, 0.5], [0, 0]]
        ]]

        # Indices are in (y, x) order.
        # As the pixels are splatted on at exact integer coordinates, there will be duplicates.
        expected_indices = [[
            [0, 0], [0, 0], [1, 0], [1, 0],
            [0, 1], [0, 1], [0, 1], [0, 1],
            [1, 0], [1, 0], [1, 1], [1, 1],
            [1, 1], [1, 1], [1, 1], [1, 1],
        ]]

        # We expect the duplicates to not splat anything.
        expected_values = [[
            [0.5, 0], [0, 0], [0.5, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0.5], [0, 0], [0, 0.5], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-1).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features,
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-1).tolist()
        self.assertCountEqual(expected, predicted)

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
