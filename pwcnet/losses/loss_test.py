import numpy as np
import tensorflow as tf
import unittest
from pwcnet.losses.loss import create_multi_level_loss, create_multi_level_unflow_loss


class TestPWCNetLosses(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_multi_level_loss(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 2
        width = 2
        batch_size = 2
        scaling = 2.0

        # Create the graph.
        expected_tensor = tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32)
        flow_tensors = [tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32),
                        tf.placeholder(shape=[None, height / 2, width / 2, 2], dtype=tf.float32)]
        flow_layer_loss_weights = [1.0, 0.5]

        expected = np.ones(shape=[batch_size, height, width, 2], dtype=np.float32)
        flows = [np.ones(shape=[batch_size, height, width, 2], dtype=np.float32) * scaling,
                 np.ones(shape=[batch_size, int(height / 2), int(width / 2), 2], dtype=np.float32) * scaling]
        flows[0][:, 0, 0, 0] = 0.0
        flows[1][:, 0, 0, 1] = 0.0

        total_loss, layer_losses = create_multi_level_loss(
            expected_tensor, flow_tensors, scaling, flow_layer_loss_weights)

        # Test loss values.
        results = self.sess.run([total_loss] + layer_losses, feed_dict={expected_tensor: expected,
                                                                        flow_tensors[0]: flows[0],
                                                                        flow_tensors[1]: flows[1]})
        self.assertEqual(3, len(results))
        self.assertEqual(6.0, results[0])
        self.assertEqual(4.0, results[1])
        self.assertEqual(2.0, results[2])

        # Test gradients.
        grad_ops = tf.gradients(total_loss, [expected_tensor, flow_tensors[0], flow_tensors[1]])
        for grad_op in grad_ops:
            self.assertNotEqual(None, grad_op)
        grads = self.sess.run(grad_ops, feed_dict={expected_tensor: expected,
                                                   flow_tensors[0]: flows[0],
                                                   flow_tensors[1]: flows[1]})
        self.assertTrue(np.allclose(np.asarray([[[[[4., 2.], [0., 0.]],
                                                  [[0., 0.], [0., 0.]]],
                                                 [[[4., 2.], [0., 0.]],
                                                  [[0., 0.], [0., 0.]]]]]), grads[0]))
        self.assertTrue(np.allclose(np.asarray([[[[[-2., 0.], [0., 0.]],
                                                  [[0., 0.], [0., 0.]]],
                                                 [[[-2., 0.], [0., 0.]],
                                                  [[0., 0.], [0., 0.]]]]]), grads[1]))
        self.assertTrue(np.allclose(np.asarray([[[[0., -1.]]],
                                                [[[0., -1.]]]]), grads[2]))

    def test_multi_level_unflow_loss(self):
        height = 8
        width = 8
        batch_size = 2
        scaling = 0.01

        # Create the graph.
        image_a_tensor = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32)
        image_b_tensor = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32)
        forward_flow_tensors = [tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32),
                                tf.placeholder(shape=[None, height / 2, width / 2, 2], dtype=tf.float32)]
        backward_flow_tensors = [tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32),
                                 tf.placeholder(shape=[None, height / 2, width / 2, 2], dtype=tf.float32)]
        flow_layer_loss_weights = [1.0, 0.5]
        layer_patch_distances = [2, 1]

        total_loss, layer_losses, _, _ = create_multi_level_unflow_loss(
            image_a_tensor, image_b_tensor, forward_flow_tensors, backward_flow_tensors, scaling,
            flow_layer_loss_weights, layer_patch_distances)

        image_a = np.ones(shape=[batch_size, height, width, 3], dtype=np.float32)
        image_b = np.ones(shape=[batch_size, height, width, 3], dtype=np.float32)
        image_a[:, :, 0, :] = 0.1
        image_b[:, 0, :, :] = 0.1
        forward_flows = [np.ones(shape=[batch_size, height, width, 2], dtype=np.float32) * scaling,
                         np.ones(shape=[batch_size, int(height / 2), int(width / 2), 2], dtype=np.float32) * scaling]
        backward_flows = [-np.ones(shape=[batch_size, height, width, 2], dtype=np.float32) * scaling * 0.5,
                          -np.ones(shape=[batch_size, int(height / 2), int(width / 2), 2], dtype=np.float32) * scaling]
        forward_flows[0][:, :2, 0, 0] = 0.1
        forward_flows[1][:, 0, 1:, 1] = 0.05
        backward_flows[0][:, 0, :3, 0] = 0.03
        backward_flows[1][:, :1, 0, 1] = 0.1
        feed_dict = {
            image_a_tensor: image_a, image_b_tensor: image_b,
            forward_flow_tensors[0]: forward_flows[0], forward_flow_tensors[1]: forward_flows[1],
            backward_flow_tensors[0]: backward_flows[0], backward_flow_tensors[1]: backward_flows[1]
        }

        # Test loss values.
        results = self.sess.run([total_loss] + layer_losses, feed_dict=feed_dict)
        self.assertEqual(3, len(results))
        self.assertEqual(results[0], results[1] + results[2])

        # Test gradients.
        grad_ops = tf.gradients(total_loss, [forward_flow_tensors[0], forward_flow_tensors[1],
                                             backward_flow_tensors[0], backward_flow_tensors[1],
                                             image_a_tensor, image_b_tensor])
        for grad_op in grad_ops:
            self.assertNotEqual(None, grad_op)
        grads = self.sess.run(grad_ops, feed_dict=feed_dict)
        for grad in grads:
            self.assertGreater(np.sum(np.abs(grad)), 0.0)


if __name__ == '__main__':
    unittest.main()
