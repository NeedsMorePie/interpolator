import numpy as np
import tensorflow as tf
import unittest
from pwcnet.cost_volume.cost_volume import cost_volume


class TestCostVolume(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

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

    def testTinyImageOnesLargerWidthSearch0(self):
        image_shape = (1, 2, 3, 1)
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

    def testTinyImageLargerWidthSearch1(self):
        image_shape = (1, 2, 3, 1)
        c1 = [
            [1, 2, 1],
            [2, 3, 2]
        ]
        c2 = [
            [1, 3, 1],
            [4, 5, 3]
        ]
        c1 = np.expand_dims(np.expand_dims(c1, axis=0), axis=-1)
        c2 = np.expand_dims(np.expand_dims(c2, axis=0), axis=-1)
        expected = np.array([
            [[0, 0, 0, 0, 1, 3, 0, 4, 5], [0, 0, 0, 2, 6, 2, 8, 10, 6], [0, 0, 0, 3, 1, 0, 5, 3, 0]],
            [[0, 2, 6, 0, 8, 10, 0, 0, 0], [3, 9, 3, 12, 15, 9, 0, 0, 0], [6, 2, 0, 10, 6, 0, 0, 0, 0]]
        ]).astype(np.float32)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())

    def testTinyImageLargerHeightSearch1(self):
        image_shape = (1, 3, 2, 1)
        c1 = [
            [1, 2],
            [2, 3],
            [1, 1]
        ]
        c2 = [
            [1, 3],
            [0, 1],
            [1, 0]
        ]
        c1 = np.expand_dims(np.expand_dims(c1, axis=0), axis=-1)
        c2 = np.expand_dims(np.expand_dims(c2, axis=0), axis=-1)
        expected = np.array([
            [[0, 0, 0, 0, 1, 3, 0, 0, 1], [0, 0, 0, 2, 6, 0, 0, 2, 0]],
            [[0, 2, 6, 0, 0, 2, 0, 2, 0], [3, 9, 0, 0, 3, 0, 3, 0, 0]],
            [[0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0]]
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
        expected = np.repeat(expected, 2, axis=0)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())

    def testTinyImageSearch1BatchOnes(self):
        image_shape = (2, 2, 2, 1)
        c1 = [[1, 2], [2, 3]]
        c2 = [[1, 3], [4, 5]]
        c1 = np.expand_dims(c1, axis=-1)
        c2 = np.expand_dims(c2, axis=-1)

        ones_shape = (2, 2, 1)
        ones1 = np.ones(ones_shape)
        ones2 = np.ones(ones_shape)

        c1 = np.stack([c1, ones1])
        c2 = np.stack([c2, ones2])

        expected = np.array([
            [
                [[0, 0, 0, 0, 1, 3, 0, 4, 5], [0, 0, 0, 2, 6, 0, 8, 10, 0]],
                [[0, 2, 6, 0, 8, 10, 0, 0, 0], [3, 9, 0, 12, 15, 0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0, 1, 1, 0]],
                [[0, 1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0, 0]]
            ]
        ]).astype(np.float32)

        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())

    def testSmallImageSearch1(self):
        image_shape = (1, 3, 3, 2)
        c1 = [
            [[0, 0], [2, 3], [5, 3]],
            [[3, 0], [0, 0], [7, 4]],
            [[0, 6], [1, 0], [5, 8]]
        ]
        c2 = [
            [[4, 2], [2, 0], [6, 2]],
            [[-3, 5], [3, 1], [0, 2]],
            [[0, 1], [1, 0], [0, 0]]
        ]
        c1, c2 = np.expand_dims(c1, axis=0), np.expand_dims(c2, axis=0)
        expected = np.array([
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 7, 2, 9, 4.5, 4.5, 3], [0, 0, 0, 5, 18, 0, 9, 3, 0]],
            [[0, 6, 3, 0, -4.5, 4.5, 0, 0, 1.5], [0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 25, 0, 12.5, 4, 0, 3.5, 0, 0]],
            [[0, 15, 3, 0, 3, 0, 0, 0, 0], [-1.5, 1.5, 0, 0, 0.5, 0, 0, 0, 0], [11.5, 8, 0, 2.5, 0, 0, 0, 0, 0]]
        ]).astype(np.float32)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(np.squeeze(cv).tolist(), expected.tolist())


if __name__ == '__main__':
    unittest.main()
