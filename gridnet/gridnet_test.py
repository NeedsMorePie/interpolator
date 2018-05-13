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

        # Note that test might fail if you change one of these parameters without modifying the expected values.
        self.num_input_channels = 8
        self.num_channels = [self.num_input_channels, 16, 32]
        self.grid_height = len(self.num_channels)
        self.grid_width = 4
        self.num_lateral_convs_per_connection = 1
        self.num_downsampling_convs_per_connection = 3
        self.num_upsampling_convs_per_connection = 3
        self.gridnet = GridNet(self.num_channels,
                               self.grid_width,
                               num_lateral_convs=self.num_lateral_convs_per_connection,
                               num_downsampling_convs=self.num_downsampling_convs_per_connection,
                               num_upsample_convs=self.num_upsampling_convs_per_connection,
                               regularizer=l2_regularizer(1e-4))

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """

        height = 32
        width = 60
        num_features = self.num_input_channels
        batch_size = 3

        # Create the graph.
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        final_output, grid_outputs = self.gridnet.get_forward(input_features_tensor)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 4:height-4, 5:width-5, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [final_output, grid_outputs]
        final_output_np, grid_outputs_np = self.sess.run(query, feed_dict={input_features_tensor: input_features})

        # Check final output shape.
        self.assertTrue(np.allclose(final_output_np.shape, np.asarray([batch_size, height, width, self.num_input_channels])))

        # Check the number of grid outputs.
        num_grid_nodes = len(grid_outputs_np) * len(grid_outputs_np[0])
        self.assertEqual(num_grid_nodes, self.grid_width * self.grid_height)

        # Check grid output shapes.
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                sample_factor = np.power(2.0, -i)
                row_num_channels = self.num_channels[i]
                expected_shape = np.asarray([batch_size, height * sample_factor, width * sample_factor, row_num_channels])
                self.assertTrue(np.allclose(grid_outputs_np[i][j].shape, expected_shape))

        # Check grid output values.
        for i in range(len(grid_outputs_np)):
            for j in range(len(grid_outputs_np[i])):
                self.assertNotEqual(np.sum(grid_outputs_np[i][j]), 0.0)

        # Test regularization losses. The magic numbers here have to do with grid net width and height.
        num_total_convs = 0
        num_total_convs += self.num_upsampling_convs_per_connection * 4
        num_total_convs += self.num_downsampling_convs_per_connection * 4
        num_total_convs += self.num_lateral_convs_per_connection * 11

        # Num conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope='gridnet')
        self.assertEqual(len(reg_losses), num_total_convs * 2)

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope='gridnet')
        self.assertEqual(len(trainable_vars), num_total_convs * 3)
        self.assertEqual(trainable_vars[1].name, 'gridnet/right_00/conv_0/kernel:0')

        # Check that the gradients are flowing.
        grad_op = tf.gradients(final_output,
                               trainable_vars + [input_features_tensor])
        gradients = self.sess.run(grad_op, feed_dict={input_features_tensor: input_features})
        for gradient in gradients:
            print(gradient)
            self.assertNotEqual(np.sum(gradient), 0.0)
