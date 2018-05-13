import numpy as np
import tensorflow as tf
import unittest
from gridnet.gridnet import GridNet
from tensorflow.contrib.layers import l2_regularizer


class TestGridNet(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.gridnet = GridNet(2, 3, [8, 16, 32],
        #                        num_lateral_convs=1,
        #                        num_downsample_convs=3,
        #                        num_upsample_convs=3,
        #                        regularizer=l2_regularizer(1e-4))

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """

        # height = 32
        # width = 60
        # num_features = 32
        # batch_size = 3
        #
        # # Create the graph.
        # input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        # final_flow, layer_outputs = self.gridnet.get_forward(input_features_tensor)
        #
        # input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        # input_features[:, 4:height-4, 5:width-5, :] = 1.0
        # input_flow = np.ones(shape=[batch_size, height, width, 2], dtype=np.float32)
        #
        # self.sess.run(tf.global_variables_initializer())
        #
        # query = [final_flow] + layer_outputs
        # results = self.sess.run(query, feed_dict={input_features1_tensor: input_features1,
        #                                           input_features2_tensor: input_features2,
        #                                           input_flow_tensor: input_flow})
        #
        # self.assertEqual(len(results), 9)
        #
        # final_flow_result = results[0]
        # self.assertTrue(np.allclose(final_flow_result.shape, np.asarray([batch_size, height, width, 2])))
        #
        # # Test that the default values are working.
        # self.assertTrue(np.allclose(results[1].shape, np.asarray([batch_size, height, width, 32])))
        # self.assertTrue(np.allclose(results[2].shape, np.asarray([batch_size, height, width, 81])))
        # self.assertTrue(np.allclose(results[3].shape, np.asarray([batch_size, height, width, 128])))
        # self.assertTrue(np.allclose(results[4].shape, np.asarray([batch_size, height, width, 128])))
        # self.assertTrue(np.allclose(results[5].shape, np.asarray([batch_size, height, width, 96])))
        # self.assertTrue(np.allclose(results[6].shape, np.asarray([batch_size, height, width, 64])))
        # self.assertTrue(np.allclose(results[7].shape, np.asarray([batch_size, height, width, 32])))
        # self.assertTrue(np.allclose(results[8].shape, np.asarray([batch_size, height, width, 2])))
        #
        # for i in range(1, 9):
        #     self.assertNotEqual(np.sum(results[i]), 0.0)
        #
        # # Test regularization losses.
        # # 6 conv layers x2 (bias and kernels).
        # reg_losses = tf.losses.get_regularization_losses(scope='estimator_network')
        # self.assertEqual(len(reg_losses), 12)
        # # Make sure the reg losses aren't 0.
        # reg_loss_sum_tensor = tf.add_n(reg_losses)
        # reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        # self.assertNotEqual(reg_loss_sum, 0.0)
        #
        # # Test that we have all the trainable variables.
        # trainable_vars = tf.trainable_variables(scope='estimator_network')
        # self.assertEqual(len(trainable_vars), 12)
        # self.assertEqual(trainable_vars[2].name, 'estimator_network/conv_1/kernel:0')