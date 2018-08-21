import numpy as np
import tensorflow as tf
import unittest
import time
from pwcnet.cost_volume.cost_volume import cost_volume
from tensorflow.python.ops import gradient_checker


PROFILE = False


class TestCostVolume(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Inputs should be normalized in some manner, i.e in range [0, 1].
        self.max_allowable_grad_err = 1E-4

    def test_tiny_image_search_0(self):
        image_shape = (1, 2, 2, 1)
        c1 = np.zeros(image_shape)
        c2 = np.ones(image_shape)
        expected = np.zeros(image_shape)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 0)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(cv.tolist(), expected.tolist())

    def test_tiny_image_ones_search_0(self):
        image_shape = (1, 2, 2, 1)
        c1 = np.ones(image_shape)
        expected = np.ones(image_shape)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 0)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c1})
        self.assertEqual(cv.tolist(), expected.tolist())

    def test_tiny_image_ones_larger_width_search_0(self):
        image_shape = (1, 2, 3, 1)
        c1 = np.ones(image_shape)
        expected = np.ones(image_shape)
        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 0)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c1})
        self.assertEqual(cv.tolist(), expected.tolist())

    def test_tiny_image_search_1(self):
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

    def test_tiny_image_larger_width_search_1(self):
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

    def test_tiny_image_larger_height_search_1(self):
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

    def test_tiny_image_search_1_batch(self):
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

    def test_tiny_image_search_1_batch_one(self):
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

    def test_small_image_search_1(self):
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

    def test_small_image_search_2(self):
        image_shape = (1, 2, 3, 4)
        c1 = [
            [[0, 0, 1, 1], [2, 3, -2, 1], [5, 3, 1, 2]],
            [[3, 0, 1, 3], [0, 0, 3, 1], [7, 4, 0, 1]],
        ]
        c2 = [
            [[4, 2, 5, 2], [2, 0, 2, 1], [6, 2, 1, -1]],
            [[-3, 5, -10, 1], [3, 1, -2, 0], [0, 2, 10, 3]],
        ]
        c1, c2 = np.expand_dims(c1, axis=0), np.expand_dims(c2, axis=0)
        expected = np.array([[[[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
             0.  ,  0.  ,  0.  ,  1.75,  0.75,  0.  ,  0.  ,  0.  , -2.25,
            -0.5 ,  3.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
             0.  ,  0.  ,  1.5 ,  0.25,  3.75,  0.  ,  0.  ,  7.5 ,  3.25,
            -2.75,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
             0.  ,  8.75,  3.5 ,  8.75,  0.  ,  0.  , -2.  ,  4.  ,  5.5 ,
             0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]],
          [[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  5.75,  2.75,
             4.  ,  0.  ,  0.  , -4.  ,  1.75,  4.75,  0.  ,  0.  ,  0.  ,
             0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  4.25,  1.75,  0.5 ,
             0.  ,  0.  , -7.25, -1.5 ,  8.25,  0.  ,  0.  ,  0.  ,  0.  ,
             0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  9.5 ,  3.75, 12.25,  0.  ,
             0.  ,  0.  ,  6.25,  2.75,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
             0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]]]])

        input1 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        input2 = tf.placeholder(shape=image_shape, dtype=tf.float32)
        cv = cost_volume(input1, input2, 2)
        cv = self.sess.run(cv, feed_dict={input1: c1, input2: c2})
        self.assertEqual(cv.tolist(), expected.tolist())

    def test_gradient_small_image_search_0(self):
        image_shape = (1, 2, 2, 1)
        c1 = np.zeros(image_shape)
        c2 = np.ones(image_shape)
        input1 = tf.constant(c1, dtype=tf.float32)
        input2 = tf.constant(c2, dtype=tf.float32)
        cv = cost_volume(input1, input2, 0)

        with self.sess:
            cv_shape = [1, 2, 2, 1]
            err_c1 = gradient_checker.compute_gradient_error(input1, image_shape, cv, cv_shape, x_init_value=c1)
            err_c2 = gradient_checker.compute_gradient_error(input2, image_shape, cv, cv_shape, x_init_value=c2)
            self.assertLessEqual(err_c1, self.max_allowable_grad_err)
            self.assertLessEqual(err_c2, self.max_allowable_grad_err)

    def test_gradient_tiny_image_larger_height_search_1(self):
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
        c1 = np.expand_dims(np.expand_dims(c1, axis=0), axis=-1) / np.max(c1)
        c2 = np.expand_dims(np.expand_dims(c2, axis=0), axis=-1) / np.max(c2)
        input1 = tf.constant(c1, dtype=tf.float32)
        input2 = tf.constant(c2, dtype=tf.float32)
        cv = cost_volume(input1, input2, 1)

        with self.sess:
            cv_shape = [1, 3, 2, 9]
            err_c1 = gradient_checker.compute_gradient_error(input1, image_shape, cv, cv_shape, x_init_value=c1)
            err_c2 = gradient_checker.compute_gradient_error(input2, image_shape, cv, cv_shape, x_init_value=c2)
            self.assertLessEqual(err_c1, self.max_allowable_grad_err)
            self.assertLessEqual(err_c2, self.max_allowable_grad_err)

    def test_gradient_small_image_search_2_batch(self):
        image_shape = (4, 3, 2, 1)
        c1 = np.random.rand(*image_shape)
        c2 = np.random.rand(*image_shape)
        input1 = tf.constant(c1, dtype=tf.float32)
        input2 = tf.constant(c2, dtype=tf.float32)
        cv = cost_volume(input1, input2, 2)

        with self.sess:
            cv_shape = [4, 3, 2, 25]
            err_c1 = gradient_checker.compute_gradient_error(input1, image_shape, cv, cv_shape, x_init_value=c1)
            err_c2 = gradient_checker.compute_gradient_error(input2, image_shape, cv, cv_shape, x_init_value=c2)
            self.assertLessEqual(err_c1, self.max_allowable_grad_err)
            self.assertLessEqual(err_c2, self.max_allowable_grad_err)


if __name__ == '__main__':
    unittest.main()