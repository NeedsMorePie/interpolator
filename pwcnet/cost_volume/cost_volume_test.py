import numpy as np
import tensorflow as tf
import unittest
from pwcnet.warp.spacial_transformer_network.transformer import spatial_transformer_network
from pwcnet.cost_volume.cost_volume import cost_volume

class TestCostVolume(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def testTinyImageSearch0(self):
        image_shape = (1, 2, 2, 1)
        c1 = np.zeros(image_shape)
        c2 = np.ones(image_shape)
        expected = np.zeros(image_shape)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 0)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(cv.tolist(), expected.tolist())

    def testTinyImageOnesSearch0(self):
        image_shape = (1, 2, 2, 1)
        c1 = np.ones(image_shape)
        expected = np.ones(image_shape)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 0)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c1})
        self.assertEqual(cv.tolist(), expected.tolist())

    def testTinyImageSearch1(self):
        image_shape = (1, 2, 2, 1)
        c1 = [[1, 2], [2, 3]]
        c2 = [[1, 3], [4, 5]]
        c1 = np.expand_dims(np.expand_dims(c1, axis=0), axis=-1)
        c2 = np.expand_dims(np.expand_dims(c2, axis=0), axis=-1)
        expected = np.array([
            [[0, 0, 0, 0, 1, 3, 0, 4, 5], [0, 0, 0, 2, 6, 0, 8, 10, 0]],
            [[0, 2, 6, 0, 8, 10, 0, 0, 0], [3, 9, 0, 12, 15, 0, 0, 0, 0]]
        ]).astype(np.float32)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())

    def testTinyImageSearch1Batch(self):
        image_shape = (2, 2, 2, 1)
        c1 = [[1, 2], [2, 3]]
        c2 = [[1, 3], [4, 5]]
        c1 = np.expand_dims(np.repeat(np.expand_dims(c1, axis=0), 2, axis=0), axis=-1)
        c2 = np.expand_dims(np.repeat(np.expand_dims(c2, axis=0), 2, axis=0), axis=-1)
        expected = np.array([
            [[0, 0, 0, 0, 1, 3, 0, 4, 5], [0, 0, 0, 2, 6, 0, 8, 10, 0]],
            [[0, 2, 6, 0, 8, 10, 0, 0, 0], [3, 9, 0, 12, 15, 0, 0, 0, 0]]
        ]).astype(np.float32)
        expected = np.expand_dims(expected, axis=0)
        expected = np.concatenate([expected, expected], axis=0)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())

    """
    def testSearch0SmallImage(self):
        image_shape = (1, 2, 2, 2)
        c1 = [
            [[1, 3], [2, 4]],
            [[2, 6], [3, 10]]
        ]
        c2 = [
            [[1, 3], [3, 5]],
            [[4, 4], [5, 5]]
        ]
        c1 = np.expand_dims(c1, axis=0)
        c2 = np.expand_dims(c2, axis=0)
        expected = np.array([
            [[0, 0, 0, 0, 1, 3, 0, 4, 5], [0, 0, 0, 2, 6, 0, 8, 10, 0]],
            [[0, 2, 6, 0, 8, 10, 0, 0, 0], [3, 9, 0, 12, 15, 0, 0, 0, 0]]
        ]).astype(np.float32)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())
    

    def testSearch1SmallImage(self):
"""
#
#
# class TestSpacialTransformTranslate(unittest.TestCase):
#     def setUp(self):
#         # Image dimensions.
#         self.height = 70
#         self.width = 90
#         self.channels = 3
#         self.image_shape = (1, self.height, self.width, self.channels)
#
#         # Defines how many pixels to translate by.
#         trans_amount = 20
#
#         # Defines the square position and size.
#         y_start = 10
#         y_end = 40
#         x_start = 20
#         x_end = 50
#
#         # The maximum diff is the circumference of the square.
#         self.max_diff = (x_end - x_start) * 2 + (y_end - y_start) * 2
#
#         # Input.
#         self.box_image = np.zeros(shape=self.image_shape, dtype=np.float)
#         self.box_image[:, y_start:y_end, x_start:x_end, :] = 1.0
#
#         # Expected output.
#         self.translated_box_image = np.zeros(shape=self.image_shape, dtype=np.float)
#         self.translated_box_image[:, y_start+trans_amount:y_end+trans_amount,
#                                   x_start+trans_amount:x_end+trans_amount, :] = 1.0
#
#         # Warp matrix to achieve the expected output.
#         self.warp = np.asarray([[1.0, 0.0, -float(trans_amount) / self.width * 2.0,
#                                  0.0, 1.0, -float(trans_amount) / self.height * 2.0]], dtype=np.float)
#
#         self.sess = tf.Session()
#
#     def runTest(self):
#         input = tf.placeholder(shape=self.image_shape, dtype=tf.float32)
#         theta = tf.placeholder(shape=[self.image_shape[0], 6], dtype=tf.float32)
#         transformed = spatial_transformer_network(input, theta)
#
#         # Get the resulting image.
#         transformed_image = self.sess.run(transformed, feed_dict={input: self.box_image, theta: self.warp})
#
#         # Get the diff.
#         diff_img = np.abs(transformed_image - self.translated_box_image)
#         diff = np.sum(np.mean(diff_img, axis=-1))
#
#         self.assertLess(diff, self.max_diff)


if __name__ == '__main__':
    unittest.main()
