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
from pylab import savefig

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
        expected_indices = [[
            [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
            [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
            [1, 1], [0, 1], [2, 1], [1, 0], [1, 2],
            [1, 1], [0, 1], [2, 1], [1, 0], [1, 2]
        ]]

        expected_values = [[
            [1, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-2).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features, 
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-2).tolist()
        self.assertCountEqual(predicted, expected)

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
        expected_indices = [[
            [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
            [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
            [1, 2], [0, 2], [2, 2], [1, 1], [1, 3],
            [1, 1], [0, 1], [2, 1], [1, 0], [1, 2],
            [1, 1], [0, 1], [2, 1], [1, 0], [1, 2],
            [1, 2], [0, 2], [2, 2], [1, 1], [1, 3]
        ]]

        # We expect the duplicates to not splat anything.
        expected_values = [[
            [1, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-2).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features,
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-2).tolist()
        self.assertCountEqual(predicted, expected)

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
        expected_indices = [[
            [0, 0], [0, 0], [1, 0], [1, 0], [0, 0],
            [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
            [1, 0], [1, 0], [1, 1], [1, 1], [1, 0],
            [1, 1], [0, 1], [2, 1], [1, 0], [1, 2]
        ]]

        # We expect the duplicates to not splat anything.
        expected_values = [[
            [0.5, 0], [0, 0], [0.5, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0.5], [0, 0], [0, 0.5], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-2).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features,
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-2).tolist()
        self.assertCountEqual(predicted, expected)

    def test_push_pixels_partial_2(self):
        height = 2
        width = 2
        features = [[
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]]
        ]]

        # Translation is in (y, x) order.
        translations = [[
            [[0.5, 0.5], [0, 0]],
            [[0, -1], [0, 0]]
        ]]

        # Indices are in (y, x) order.
        expected_indices = [[
            [0, 0], [0, 1], [1, 0], [1, 1], [0, 0],
            [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
            [1, -1], [0, -1], [2, -1], [1, -2], [1, 0],
            [1, 1], [0, 1], [2, 1], [1, 0], [1, 2]
        ]]

        # We expect the duplicates to not splat anything.
        expected_values = [[
            [0.25, 0], [0.25, 0], [0.25, 0], [0.25, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
        ]]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-2).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[1, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features,
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-2).tolist()
        self.assertCountEqual(predicted, expected)

    def test_push_pixels_batch(self):
        height = 2
        width = 2
        features = [
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]]
            ],
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]]
            ]
        ]

        # Translation is in (y, x) order.
        # Translates first column to second column.
        translations = [
            [
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]]
            ],
            [
                [[0.5, 0], [0, 0]],
                [[0, 0.5], [0, 0]]
            ]
        ]

        # Indices are in (y, x) order.
        # As the pixels are splatted on at exact integer coordinates, there will be duplicates.
        expected_indices = [
            [
                [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
                [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
                [1, 1], [0, 1], [2, 1], [1, 0], [1, 2],
                [1, 1], [0, 1], [2, 1], [1, 0], [1, 2]
            ],
            [
                [0, 0], [0, 0], [1, 0], [1, 0], [0, 0],
                [0, 1], [-1, 1], [1, 1], [0, 0], [0, 2],
                [1, 0], [1, 0], [1, 1], [1, 1], [1, 0],
                [1, 1], [0, 1], [2, 1], [1, 0], [1, 2]
            ]
        ]

        # We expect the duplicates to not splat anything.
        expected_values = [
            [
                [1, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
            ],
            [
                [0.5, 0], [0, 0], [0.5, 0], [0, 0], [0, 0],
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, 0.5], [0, 0], [0, 0.5], [0, 0], [0, 0],
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
            ]
        ]

        expected_indices, expected_values = np.squeeze(expected_indices), np.squeeze(expected_values)
        expected = np.stack([expected_indices, expected_values], axis=-2).tolist()

        features_tensor = tf.placeholder(tf.float32, shape=[2, height, width, 2])
        translations_tensor = tf.placeholder(tf.float32, shape=[2, height, width, 2])
        indices_tensor, values_tensor = get_translated_pixels(features_tensor, translations_tensor)

        query = [indices_tensor, values_tensor]
        indices, values = self.sess.run(query, feed_dict={features_tensor: features,
                                                          translations_tensor: translations})

        indices, values = indices.tolist(), values.tolist()
        indices, values = np.squeeze(indices), np.squeeze(values)
        predicted = np.stack([indices, values], axis=-2).tolist()
        self.assertCountEqual(predicted[0], expected[0])
        self.assertCountEqual(predicted[1], expected[1])

    def test_forward_warp_whole_1(self):
        height = 2
        width = 2

        # Flow is in (x, y) order.
        # Splats the top-left pixel right in the center.
        flow = [[
            [[1, 1], [0, 0]],
            [[0, 0], [0, 0]]
        ]]
        features = [[
            [[4, 0], [0, 0]],
            [[1, 1], [0, 0]]
        ]]
        expected_warp = [[
            [[0, 0], [0, 0]],
            [[1, 1], [4, 0]]
        ]]

        flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        warp_tensor = forward_warp(features_tensor, flow_tensor)
        warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
        self.assertEqual(warp.tolist(), expected_warp)

    def test_forward_warp_partial_1(self):
        height = 2
        width = 2

        # Flow is in (x, y) order.
        # Splats the top-left pixel right in the center.
        flow = [[
            [[0.5, 0.5], [0, 0]],
            [[0, 0], [0, 0]]
        ]]
        features = [[
            [[4, 0], [0, 0]],
            [[1, 1], [0, 0]]
        ]]
        expected_warp = [[
            [[1, 0], [1, 0]],
            [[2, 1], [1, 0]]
        ]]

        flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        warp_tensor = forward_warp(features_tensor, flow_tensor)
        warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
        self.assertEqual(warp.tolist(), expected_warp)

    def test_forward_warp_partial_2(self):
        height = 3
        width = 2

        # Flow is in (x, y) order.
        # Splats the top-left pixel right in the center.
        flow = [[
            [[0.5, 0.5], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [-0.5, -0.5]]
        ]]
        features = [[
            [[4, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[0, 0], [-4, -4]]
        ]]
        expected_warp = [[
            [[1, 0], [1, 0]],
            [[1, 0], [0, -1]],
            [[-1, -1], [-1, -1]]
        ]]

        flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        warp_tensor = forward_warp(features_tensor, flow_tensor)
        warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
        self.assertEqual(warp.tolist(), expected_warp)

        # Check for gradients.
        grads_tensor = tf.gradients(warp_tensor[0][0][0], [flow_tensor, features_tensor])
        for grad_tensor in grads_tensor:
            self.assertNotEqual(grad_tensor, None)

        # For the top left warp, there should only be 2 flows giving non-zero gradients.
        grads = self.sess.run(grads_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
        flow_grads, feature_grads = grads[0][0], grads[1][0]
        self.assertNotEqual(np.sum(flow_grads[0][0]), 0.0)
        self.assertNotEqual(np.sum(flow_grads[1][0]), 0.0)
        self.assertEqual(np.sum(flow_grads) - np.sum(flow_grads[0][0]) - np.sum(flow_grads[1][0]), 0.0)

        # Note that the feature on the left edge (at [1][0]) should have 0 gradient.
        self.assertNotEqual(np.sum(feature_grads[0][0]), 0.0)
        self.assertEqual(np.sum(feature_grads) - np.sum(feature_grads[0][0]), 0.0)

    def test_forward_warp_oob(self):
        """
        Note that oob == out of bounds.
        """
        height = 3
        width = 2

        # Flow is in (x, y) order.
        # Splats the top-left pixel right in the center.
        flow = [[
            [[1.5, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [-10, -10]]
        ]]
        features = [[
            [[4, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[0, 0], [-4, -4]]
        ]]
        expected_warp = [[
            [[0, 0], [2, 0]],
            [[1, 1], [0, 0]],
            [[0, 0], [0, 0]]
        ]]

        flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
        warp_tensor = forward_warp(features_tensor, flow_tensor)
        warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
        self.assertEqual(warp.tolist(), expected_warp)

    def test_forward_warp_batch(self):
        height = 2
        width = 2

        # Flow is in (x, y) order.
        # Splats the top-left pixel right in the center.
        flow = [
            [
                [[0.5, 0.5], [0, 0]],
                [[0, 0], [0, 0]]
            ],
            [
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]]
            ]
        ]
        features = [
            [
                [[4, 0], [0, 0]],
                [[1, 1], [0, 0]]
            ],
            [
                [[100, 0], [0, 0]],
                [[1, 1], [0, 0]]
            ]
        ]
        expected_warp = [
            [
                [[1, 0], [1, 0]],
                [[2, 1], [1, 0]]
            ],
            [
                [[0, 0], [0, 0]],
                [[1, 1], [100, 0]]
            ]
        ]

        flow_tensor = tf.placeholder(tf.float32, (2, height, width, 2))
        features_tensor = tf.placeholder(tf.float32, (2, height, width, 2))
        warp_tensor = forward_warp(features_tensor, flow_tensor)
        warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
        self.assertEqual(warp[0].tolist(), expected_warp[0])
        self.assertEqual(warp[1].tolist(), expected_warp[1])

    def test_visualization(self):
        if not VISUALIZE:
            return

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(cur_dir, '..')
        flow_path = os.path.join(root_dir, 'pwcnet', 'warp', 'test_data', 'flow_ab.flo')
        image_path = os.path.join(root_dir, 'pwcnet', 'warp', 'test_data', 'image_a.png')

        flow_ab = [read_flow_file(flow_path)]
        img_a = [read_image(image_path, as_float=True)]
        t_tensor = tf.placeholder(tf.float32, None)
        flow_ab_tensor = tf.placeholder(tf.float32, np.shape(flow_ab))
        img_a_tensor = tf.placeholder(tf.float32, np.shape(img_a))
        warp_tensor = forward_warp(img_a_tensor, t_tensor * flow_ab_tensor)

        warp = self.sess.run(warp_tensor, feed_dict={flow_ab_tensor: flow_ab, img_a_tensor: img_a, t_tensor: 1.0})
        warp = np.clip(warp[0], 0.0, 1.0)
        show_image(warp)

        # For writing to video.
        # height = img_a[0].shape[0]
        # width = img_a[0].shape[1]
        # writer = cv2.VideoWriter(cur_dir + '/outputs/video-jank.avi', cv2.VideoWriter_fourcc(*"MJPG"), 20, (width, height))
        #
        # steps = 60
        # for i in range(steps):
        #     print('Writing video at step %d' % i)
        #     t = i * (1.0 / float(steps))
        #     warped = self.sess.run(warp_tensor,
        #                            feed_dict={flow_ab_tensor: flow_ab, img_a_tensor: img_a, t_tensor: t})
        #     warped = warped[0]
        #     warped = np.clip(warped, 0.0, 1.0)
        #     output_path = cur_dir + "/outputs/out-%.2f.png" % t
        #     mpimg.imsave(output_path, warped)
        #     writer.write(cv2.imread(output_path))
        #
        # writer.release()
