import numpy as np
import unittest
from common.utils.multi_gpu import *


class TestMultiGPUUtils(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto(device_count={'CPU': 2})
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

    def test_accumulating_tensor_io_empty(self):
        accumulating_tensor_io = AccumulatingTensorIO(TensorIO.AVERAGED_SCALAR)
        accumulated_io = accumulating_tensor_io.accumulate()
        self.assertTrue(isinstance(accumulated_io, TensorIO))
        self.assertEqual(TensorIO.AVERAGED_SCALAR, accumulated_io.tensor_type)
        self.assertEqual(0, len(accumulated_io.tensors))

    def test_accumulating_tensor_io_averaged_scalar(self):
        scalar_0_1 = tf.constant(1.0, dtype=tf.float32)
        scalar_0_2 = tf.constant(2.0, dtype=tf.float32)
        scalar_1_1 = tf.constant(3.0, dtype=tf.float32)
        scalar_1_2 = tf.constant(7.0, dtype=tf.float32)
        accumulating_tensor_io = AccumulatingTensorIO(TensorIO.AVERAGED_SCALAR)
        accumulating_tensor_io.add([scalar_0_1, scalar_1_1])
        accumulating_tensor_io.add([scalar_0_2, scalar_1_2])
        accumulated_io = accumulating_tensor_io.accumulate()
        self.assertTrue(isinstance(accumulated_io, TensorIO))
        self.assertEqual(TensorIO.AVERAGED_SCALAR, accumulated_io.tensor_type)
        outputs = self.sess.run(accumulated_io.tensors)
        self.assertEqual(2, len(outputs))
        self.assertEqual(1.5, outputs[0])
        self.assertEqual(5, outputs[1])

    def test_accumulating_tensor_io_summed_scalar(self):
        scalar_0_1 = tf.constant(1.0, dtype=tf.float32)
        scalar_0_2 = tf.constant(2.0, dtype=tf.float32)
        scalar_1_1 = tf.constant(3.0, dtype=tf.float32)
        scalar_1_2 = tf.constant(7.0, dtype=tf.float32)
        accumulating_tensor_io = AccumulatingTensorIO(TensorIO.SUMMED_SCALAR)
        accumulating_tensor_io.add([scalar_0_1, scalar_1_1])
        accumulating_tensor_io.add([scalar_0_2, scalar_1_2])
        accumulated_io = accumulating_tensor_io.accumulate()
        self.assertTrue(isinstance(accumulated_io, TensorIO))
        self.assertEqual(TensorIO.SUMMED_SCALAR, accumulated_io.tensor_type)
        outputs = self.sess.run(accumulated_io.tensors)
        self.assertEqual(2, len(outputs))
        self.assertEqual(3, outputs[0])
        self.assertEqual(10, outputs[1])

    def test_accumulating_tensor_io_batched_scalar(self):
        scalar_0_1 = tf.constant([1.0], dtype=tf.float32)
        scalar_0_2 = tf.constant([2.0], dtype=tf.float32)
        scalar_1_1 = tf.constant([3.0], dtype=tf.float32)
        scalar_1_2 = tf.constant([7.0], dtype=tf.float32)
        accumulating_tensor_io = AccumulatingTensorIO(TensorIO.BATCH)
        accumulating_tensor_io.add([scalar_0_1, scalar_1_1])
        accumulating_tensor_io.add([scalar_0_2, scalar_1_2])
        accumulated_io = accumulating_tensor_io.accumulate()
        self.assertTrue(isinstance(accumulated_io, TensorIO))
        self.assertEqual(TensorIO.BATCH, accumulated_io.tensor_type)
        outputs = self.sess.run(accumulated_io.tensors)
        self.assertEqual(2, len(outputs))
        self.assertTrue(np.allclose(np.asarray([1.0, 2.0]), outputs[0]))
        self.assertTrue(np.allclose(np.asarray([3.0, 7.0]), outputs[1]))

    def test_accumulating_tensor_io_deep_batched_scalar(self):
        scalar_0_1 = tf.constant([[1.0]], dtype=tf.float32)
        scalar_0_2 = tf.constant([[2.0]], dtype=tf.float32)
        scalar_1_1 = tf.constant([[3.0]], dtype=tf.float32)
        scalar_1_2 = tf.constant([[7.0]], dtype=tf.float32)
        accumulating_tensor_io = AccumulatingTensorIO(TensorIO.BATCH)
        accumulating_tensor_io.add([scalar_0_1, scalar_1_1])
        accumulating_tensor_io.add([scalar_0_2, scalar_1_2])
        accumulated_io = accumulating_tensor_io.accumulate()
        self.assertTrue(isinstance(accumulated_io, TensorIO))
        self.assertEqual(TensorIO.BATCH, accumulated_io.tensor_type)
        outputs = self.sess.run(accumulated_io.tensors)
        self.assertEqual(2, len(outputs))
        self.assertTrue(np.allclose(np.asarray([[1.0], [2.0]]), outputs[0]))
        self.assertTrue(np.allclose(np.asarray([[3.0], [7.0]]), outputs[1]))

    def test_create_train_op_default_device(self):
        self.create_train_op_one_device_helper(None)

    def test_create_train_op_one_device(self):
        self.create_train_op_one_device_helper(['/cpu:0'])

    def create_train_op_one_device_helper(self, devices):
        def build_network_outputs(tensor_1, tensor_2, tensor_3):
            variable = tf.Variable(1.0, trainable=True, dtype=tf.float32)
            loss = tf.reduce_mean((tensor_1 + tensor_2) * tensor_3 * variable)
            return {'loss': TensorIO([loss]), 'variable': TensorIO([variable])}

        # Note this learning rate is set up such that the loss is 0.0 after 1 iteration.
        optimizer = tf.train.GradientDescentOptimizer(1.0 / 6.0)
        batched_network_args = [tf.constant([1.0, 2.0], dtype=tf.float32), tf.constant([2.0, 1.0], dtype=tf.float32)]
        other_network_args = [tf.constant(2.0, dtype=tf.float32)]

        train_op, global_step, output_dict = create_train_op(
            optimizer, build_network_outputs, batched_network_args, other_network_args, available_devices=devices)
        self.assertEqual(2, len(output_dict.keys()))
        self.assertTrue('loss' in output_dict)
        self.sess.run(tf.global_variables_initializer())
        _, loss = self.sess.run([train_op, output_dict['loss'].first()])
        self.assertEqual(6.0, loss)
        step, loss, variable = self.sess.run(
            [global_step, output_dict['loss'].first(), output_dict['variable'].first()])
        self.assertAlmostEqual(0.0, loss, places=5)
        self.assertAlmostEqual(0.0, variable, places=5)
        self.assertEqual(1, step)

    def test_create_train_op_multiple_devices(self):
        devices = self.get_devices_for_testing()
        if devices is None:
            return
        variables = []

        def build_network_outputs(tensor_1, tensor_2, tensor_3):
            with tf.variable_scope('multi_device_test_scope', reuse=tf.AUTO_REUSE):
                variable = tf.get_variable('test_variable', shape=(), dtype=tf.float32,
                                           initializer=tf.constant_initializer(1.0))
                variables.append(variable)
                loss = tf.reduce_mean((tensor_1 + tensor_2) * tensor_3 * variable)
                return {'loss': TensorIO([loss]), 'variable': TensorIO([variable])}

        # Note this learning rate is set up such that the loss is 0.0 after 1 iteration.
        optimizer = tf.train.GradientDescentOptimizer(1.0 / 6.0)
        batched_network_args = [tf.constant([1.0, 2.0], dtype=tf.float32), tf.constant([2.0, 1.0], dtype=tf.float32)]
        other_network_args = [tf.constant(2.0, dtype=tf.float32)]

        train_op, global_step, output_dict = create_train_op(
            optimizer, build_network_outputs, batched_network_args, other_network_args, available_devices=devices)
        self.assertEqual(2, len(output_dict.keys()))
        self.assertTrue('loss' in output_dict)
        self.sess.run(tf.global_variables_initializer())
        _, loss = self.sess.run([train_op, output_dict['loss'].first()])
        self.assertEqual(6.0, loss)
        step, loss, variable = self.sess.run(
            [global_step, output_dict['loss'].first(), output_dict['variable'].first()])
        self.assertAlmostEqual(0.0, loss, places=5)
        self.assertAlmostEqual(0.0, variable, places=5)
        self.assertEqual(1, step)
        variable_1, variable_2 = self.sess.run(variables)
        self.assertAlmostEqual(0.0, variable_1, places=5)
        self.assertAlmostEqual(0.0, variable_2, places=5)

    def test_create_train_op_multiple_devices_unshared_variable_failure(self):
        devices = self.get_devices_for_testing()
        if devices is None:
            return
        variables = []

        def build_network_outputs(tensor_1, tensor_2, tensor_3):
            with tf.variable_scope('multi_device_test_scope_failure', reuse=False):
                variable = tf.Variable(1.0, trainable=True, dtype=tf.float32)
                variables.append(variable)
                loss = tf.reduce_mean((tensor_1 + tensor_2) * tensor_3 * variable)
                return {'loss': TensorIO([loss]), 'variable': TensorIO([variable])}

        # Note this learning rate is set up such that the loss is 0.0 after 1 iteration.
        optimizer = tf.train.GradientDescentOptimizer(1.0 / 6.0)
        batched_network_args = [tf.constant([1.0, 2.0], dtype=tf.float32), tf.constant([2.0, 1.0], dtype=tf.float32)]
        other_network_args = [tf.constant(2.0, dtype=tf.float32)]

        train_op, global_step, output_dict = create_train_op(
            optimizer, build_network_outputs, batched_network_args, other_network_args, available_devices=devices)
        self.assertEqual(2, len(output_dict.keys()))
        self.assertTrue('loss' in output_dict)
        self.sess.run(tf.global_variables_initializer())
        _, loss = self.sess.run([train_op, output_dict['loss'].first()])
        self.assertEqual(6.0, loss)
        step, loss, variable = self.sess.run(
            [global_step, output_dict['loss'].first(), output_dict['variable'].first()])
        self.assertAlmostEqual(3.0, loss, places=5)
        self.assertAlmostEqual(0.5, variable, places=5)
        self.assertEqual(1, step)
        variable_1, variable_2 = self.sess.run(variables)
        self.assertAlmostEqual(0.0, variable_1, places=5)
        self.assertAlmostEqual(1.0, variable_2, places=5)

    def get_devices_for_testing(self):
        available_gpus = get_available_gpus()
        if len(available_gpus) == 0:
            devices = ['/cpu:0', '/cpu:1']
        elif len(available_gpus) == 1:
            devices = [available_gpus[0], '/cpu:0']
        else:
            devices = [available_gpus[0], available_gpus[1]]
        assert len(devices) == 2
        return devices


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
