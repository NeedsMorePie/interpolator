import numpy as np
import tensorflow as tf
import unittest
from pwcnet.context_network.model import ContextNetwork
from tensorflow.contrib.layers import l2_regularizer


class TestContextNetwork(unittest.TestCase):
    def setUp(self):
        self.context_network = ContextNetwork(regularizer=l2_regularizer(1e-4))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 128
        width = 200
        num_features = 32
        batch_size = 3

        # Create the graph.
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        input_flow_tensor = tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32)
        final_flow, layer_outputs = self.context_network.get_forward(input_features_tensor, input_flow_tensor)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 10:height-10, 10:width-10, :] = 1.0
        input_flow = np.ones(shape=[batch_size, height, width, 2], dtype=np.float32)

        self.sess.run(tf.global_variables_initializer())

        query = [final_flow] + layer_outputs
        results = self.sess.run(query, feed_dict={input_features_tensor: input_features,
                                                  input_flow_tensor: input_flow})

        self.assertEqual(len(results), 8)

        final_flow_result = results[0]
        self.assertTrue(np.allclose(final_flow_result.shape, np.asarray([batch_size, height, width, 2])))

        # Test that the default values are working.
        self.assertTrue(np.allclose(results[1].shape, np.asarray([batch_size, height, width, 128])))
        self.assertTrue(np.allclose(results[2].shape, np.asarray([batch_size, height, width, 128])))
        self.assertTrue(np.allclose(results[3].shape, np.asarray([batch_size, height, width, 128])))
        self.assertTrue(np.allclose(results[4].shape, np.asarray([batch_size, height, width, 96])))
        self.assertTrue(np.allclose(results[5].shape, np.asarray([batch_size, height, width, 64])))
        self.assertTrue(np.allclose(results[6].shape, np.asarray([batch_size, height, width, 32])))
        self.assertTrue(np.allclose(results[7].shape, np.asarray([batch_size, height, width, 2])))

        for i in range(1, 8):
            self.assertNotEqual(np.sum(results[i]), 0.0)

        # Test regularization losses.
        # 7 conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope='context_network')
        self.assertEqual(len(reg_losses), 14)
        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        self.assertEqual(reg_losses[0].name, 'context_network/conv_0/kernel/Regularizer/l2_regularizer:0')

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope='context_network')
        self.assertEqual(len(trainable_vars), 14)
        self.assertEqual(trainable_vars[2].name, 'context_network/conv_1/kernel:0')

        # Test that the output is indeed a sum of the delta and the input optical flow.
        delta_flow = results[7]
        reconstructed_input_flow = final_flow_result - delta_flow
        self.assertTrue(np.allclose(input_flow, reconstructed_input_flow))


if __name__ == '__main__':
    unittest.main()
