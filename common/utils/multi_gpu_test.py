import numpy as np
import unittest
from common.utils.multi_gpu import *


class TestMultiGPUUtils(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_average_gradients(self):
        grad_tensor_gpu0_1 = tf.placeholder(shape=(2, 2), dtype=tf.float32)
        grad_tensor_gpu0_2 = tf.placeholder(shape=(1, 3), dtype=tf.float32)
        grad_tensor_gpu1_1 = tf.placeholder(shape=(2, 2), dtype=tf.float32)
        grad_tensor_gpu1_2 = tf.placeholder(shape=(1, 3), dtype=tf.float32)
        dummy_var_1 = object()
        dummy_var_2 = object()

        grads_and_vars_gpu0 = [(grad_tensor_gpu0_1, dummy_var_1), (grad_tensor_gpu0_2, dummy_var_2)]
        grads_and_vars_gpu1 = [(grad_tensor_gpu1_1, dummy_var_1), (grad_tensor_gpu1_2, dummy_var_2)]

        averaged_grads_and_vars = average_gradients([grads_and_vars_gpu0, grads_and_vars_gpu1])
        self.assertEqual(2, len(averaged_grads_and_vars))
        self.assertEqual(2, len(averaged_grads_and_vars[0]))
        self.assertEqual(2, len(averaged_grads_and_vars[1]))
        self.assertEqual(dummy_var_1, averaged_grads_and_vars[0][1])
        self.assertEqual(dummy_var_2, averaged_grads_and_vars[1][1])

        average_grad_tensor_1 = averaged_grads_and_vars[0][0]
        average_grad_tensor_2 = averaged_grads_and_vars[1][0]

        feed_dict = {
            grad_tensor_gpu0_1: [
                [2, 2],
                [3, 3]
            ],
            grad_tensor_gpu0_2: [
                [3, 3, 3]
            ],
            grad_tensor_gpu1_1: [
                [4, 4],
                [1, 1]
            ],
            grad_tensor_gpu1_2: [
                [1, 3, 5]
            ]
        }
        average_grad_1, average_grad_2 = self.sess.run(
            [average_grad_tensor_1, average_grad_tensor_2], feed_dict=feed_dict)
        self.assertTrue(np.allclose(np.asarray([[3, 3],
                                                [2, 2]]), average_grad_1))
        self.assertTrue(np.allclose(np.asarray([[2, 3, 4]]), average_grad_2))

    def test_average_gradients_no_grads(self):
        grads_and_vars_gpu0 = []
        grads_and_vars_gpu1 = []
        averaged_grads_and_vars = average_gradients([grads_and_vars_gpu0, grads_and_vars_gpu1])
        self.assertListEqual([], averaged_grads_and_vars)

    def test_average_gradients_empty(self):
        averaged_grads_and_vars = average_gradients([])
        self.assertListEqual([], averaged_grads_and_vars)

    def test_average_gradients_single_gpu(self):
        grad_tensor_gpu0_1 = tf.placeholder(shape=(2, 2), dtype=tf.float32)
        grad_tensor_gpu0_2 = tf.placeholder(shape=(1, 3), dtype=tf.float32)
        dummy_var_1 = object()
        dummy_var_2 = object()
        grads_and_vars_gpu0 = [(grad_tensor_gpu0_1, dummy_var_1), (grad_tensor_gpu0_2, dummy_var_2)]
        averaged_grads_and_vars = average_gradients([grads_and_vars_gpu0])
        self.assertEqual(2, len(averaged_grads_and_vars))
        self.assertEqual(2, len(averaged_grads_and_vars[0]))
        self.assertEqual(dummy_var_1, averaged_grads_and_vars[0][1])
        self.assertEqual(dummy_var_2, averaged_grads_and_vars[1][1])

        average_grad_tensor_1 = averaged_grads_and_vars[0][0]
        average_grad_tensor_2 = averaged_grads_and_vars[1][0]

        feed_dict = {
            grad_tensor_gpu0_1: [
                [2, 2],
                [3, 3]
            ],
            grad_tensor_gpu0_2: [
                [3, 3, 3]
            ]
        }
        average_grad_1, average_grad_2 = self.sess.run(
            [average_grad_tensor_1, average_grad_tensor_2], feed_dict=feed_dict)
        self.assertTrue(np.allclose(np.asarray([[2, 2],
                                                [3, 3]]), average_grad_1))
        self.assertTrue(np.allclose(np.asarray([[3, 3, 3]]), average_grad_2))


class TestListAccumulation(unittest.TestCase):
    def test_accumulate_once(self):
        items_list = []
        items = [1, 2]
        accumulate_list_into(items, items_list)
        self.assertListEqual([[1], [2]], items_list)

    def test_accumulate_twice(self):
        items_list = []
        items = [1, 2]
        accumulate_list_into(items, items_list)
        items = [3, 4]
        accumulate_list_into(items, items_list)
        self.assertListEqual([[1, 3], [2, 4]], items_list)

    def test_empty(self):
        items_list = []
        items = []
        accumulate_list_into(items, items_list)
        self.assertListEqual([], items_list)

    def test_single(self):
        items_list = []
        items = [1]
        accumulate_list_into(items, items_list)
        items = [2]
        accumulate_list_into(items, items_list)
        self.assertListEqual([[1, 2]], items_list)


if __name__ == '__main__':
    unittest.main()
