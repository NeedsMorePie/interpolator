import numpy as np
import tensorflow as tf
import unittest
from context_interp.gridnet.model import GridNet
from tensorflow.contrib.layers import l2_regularizer


class TestGridNet(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_network(self):

        # Note that test might fail if you change one of these parameters without modifying the expected values.
        name='gridnet'
        num_input_channels = 16
        num_output_channels = 3
        num_channels = [num_input_channels, 16, 32]
        grid_height = len(num_channels)
        grid_width = 4
        num_lateral_convs_per_connection = 1
        num_downsampling_convs_per_connection = 3
        num_upsampling_convs_per_connection = 3
        gridnet = GridNet(num_channels,
                          grid_width,
                          name=name,
                          num_output_channels=num_output_channels,
                          num_lateral_convs=num_lateral_convs_per_connection,
                          num_downsampling_convs=num_downsampling_convs_per_connection,
                          num_upsample_convs=num_upsampling_convs_per_connection,
                          regularizer=l2_regularizer(1e-4))

        height = 32
        width = 60
        num_features = num_input_channels
        batch_size = 3

        # Create the graph.
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        outputs = gridnet.get_forward(input_features_tensor, training=True)
        final_output, node_outputs, lateral_inputs, vertical_inputs = outputs

        # Note that the first parametric ReLU's gradient for alpha will be 0 if inputs are all non-negative.
        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 2:height-2, 2:width-2, :] = -1.0
        input_features[:, 4:height-4, 5:width-5, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [final_output, node_outputs, lateral_inputs, vertical_inputs]
        outputs_np = self.sess.run(query, feed_dict={input_features_tensor: input_features})
        final_output_np, node_outputs_np, lateral_inputs_np, vertical_inputs_np = outputs_np

        # Check final output shape.
        self.assertTrue(np.allclose(final_output_np.shape, np.asarray([batch_size, height, width, num_output_channels])))

        # Check the number of grid outputs.
        num_grid_nodes = len(node_outputs_np) * len(node_outputs_np[0])
        self.assertEqual(num_grid_nodes, grid_width * grid_height)

        # Check grid output shapes.
        for i in range(grid_height):
            for j in range(grid_width):
                sample_factor = np.power(2.0, -i)
                row_num_channels = num_channels[i]
                expected_shape = np.asarray([batch_size, height * sample_factor, width * sample_factor, row_num_channels])
                self.assertTrue(np.allclose(node_outputs_np[i][j].shape, expected_shape))

        # Check grid node output values.
        self.assertNotEqual(np.sum(final_output_np), 0.0)
        for i in range(grid_height):
            for j in range(grid_width):
                self.assertNotEqual(np.sum(node_outputs_np[i][j]), 0.0)

        # Check stream output values.
        for i in range(grid_height):
            for j in range(grid_width):

                # Lateral inputs.
                if j > 0 or (i == 0 and j == 0):
                    self.assertNotEqual(np.sum(lateral_inputs_np[i][j]), 0.0)
                else:
                    self.assertEqual(np.sum(lateral_inputs_np[i][j]), 0.0)

                # Vertical inputs.
                if (i > 0 and j < grid_width / 2) or (i < grid_height - 1 and j >= grid_width / 2):
                    self.assertNotEqual(np.sum(vertical_inputs_np[i][j]), 0.0)
                else:
                    self.assertEqual(np.sum(vertical_inputs_np[i][j]), 0.0)

        # Check that each node is the sum of its incoming streams.
        for i in range(grid_height):
            for j in range(grid_width):
                sum = lateral_inputs_np[i][j] + vertical_inputs_np[i][j]
                self.assertEqual(sum.tolist(), node_outputs_np[i][j].tolist())

        # Test regularization losses. The magic numbers here have to do with grid net width and height.
        num_total_convs = 0
        num_total_convs += num_upsampling_convs_per_connection * 4
        num_total_convs += num_downsampling_convs_per_connection * 4
        num_total_convs += num_lateral_convs_per_connection * 11

        # Num conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope=name)
        self.assertEqual(len(reg_losses), num_total_convs * 2)

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        # Test that we have all the trainable variables.
        # Each parametric ReLU has a trainable alpha variable.
        trainable_vars = tf.trainable_variables(scope=name)
        self.assertEqual(len(trainable_vars), num_total_convs * 3)
        self.assertEqual(trainable_vars[1].name, name + '/right_0_0/conv_0/kernel:0')

        # Check that the gradients are flowing.
        grad_op = tf.gradients(final_output,
                               trainable_vars + [input_features_tensor])
        gradients = self.sess.run(grad_op, feed_dict={input_features_tensor: input_features})
        for gradient in gradients:
            self.assertNotEqual(np.sum(gradient), 0.0)

    def test_network_dropout(self):

        # Note that test might fail if you change one of these parameters without modifying the expected values.
        name='gridnet_dropped'
        num_input_channels = 8
        num_channels = [num_input_channels, 16, 32]
        grid_height = len(num_channels)
        grid_width = 4
        num_lateral_convs_per_connection = 1
        num_downsampling_convs_per_connection = 3
        num_upsampling_convs_per_connection = 3
        gridnet = GridNet(num_channels,
                          grid_width,
                          name=name,
                          connection_dropout_rate=1.0,
                          num_lateral_convs=num_lateral_convs_per_connection,
                          num_downsampling_convs=num_downsampling_convs_per_connection,
                          num_upsample_convs=num_upsampling_convs_per_connection,
                          regularizer=l2_regularizer(1e-4))

        height = 32
        width = 60
        num_features = num_input_channels
        batch_size = 3

        # Create the graph.
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        outputs = gridnet.get_forward(input_features_tensor, training=True)
        final_output, node_outputs, lateral_inputs, vertical_inputs = outputs

        # Note that the first parametric ReLU's gradient for alpha will be 0 if inputs are all non-negative.
        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 2:height-2, 2:width-2, :] = -1.0
        input_features[:, 4:height-4, 5:width-5, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [final_output, node_outputs, lateral_inputs, vertical_inputs]
        outputs_np = self.sess.run(query, feed_dict={input_features_tensor: input_features})
        final_output_np, node_outputs_np, lateral_inputs_np, vertical_inputs_np = outputs_np

        # Check final output shape.
        self.assertTrue(np.allclose(final_output_np.shape, np.asarray([batch_size, height, width, num_input_channels])))

        # Check the number of grid outputs.
        num_grid_nodes = len(node_outputs_np) * len(node_outputs_np[0])
        self.assertEqual(num_grid_nodes, grid_width * grid_height)

        # Check grid node output shapes.
        for i in range(grid_height):
            for j in range(grid_width):
                sample_factor = np.power(2.0, -i)
                row_num_channels = num_channels[i]
                expected_shape = np.asarray([batch_size, height * sample_factor, width * sample_factor, row_num_channels])
                self.assertTrue(np.allclose(node_outputs_np[i][j].shape, expected_shape))

        # Check grid output values.
        self.assertEqual(np.sum(final_output_np), 0.0)
        for i in range(grid_height):
            for j in range(grid_width):
                # The first lateral connection should always have non-zero output.
                # The first column of down-sampling of streams should also have non-zero output.
                # All other nodes have zero output when biases are zero-initialized.
                if j == 0:
                    self.assertNotEqual(np.sum(node_outputs_np[i][j]), 0.0)
                else:
                    self.assertEqual(np.sum(node_outputs_np[i][j]), 0.0)

        # Test regularization losses. The magic numbers here have to do with grid net width and height.
        num_total_convs = 0
        num_total_convs += num_upsampling_convs_per_connection * 4
        num_total_convs += num_downsampling_convs_per_connection * 4
        num_total_convs += num_lateral_convs_per_connection * 11

        # Num conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope=name)
        self.assertEqual(len(reg_losses), num_total_convs * 2)

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        # Test that we have all the trainable variables.
        # Each parametric ReLU has a trainable alpha variable.
        trainable_vars = tf.trainable_variables(scope=name)
        self.assertEqual(len(trainable_vars), num_total_convs * 3)
        self.assertEqual(trainable_vars[1].name, name + '/right_0_0/conv_0/kernel:0')

        # Check gradients.
        grad_op = tf.gradients(final_output,
                               trainable_vars + [input_features_tensor])
        gradients = self.sess.run(grad_op, feed_dict={input_features_tensor: input_features})

        nonzero_grad_names = {'up_0_3', 'up_1_3', 'right_0_4'}
        nonzero_sum = 0
        for i, gradient in enumerate(gradients):
            nonzero = False
            if i < len(trainable_vars):
                connection_name = trainable_vars[i].name.split('/')
                if len(connection_name) > 1 and connection_name[1] in nonzero_grad_names:
                    nonzero_sum += np.sum(gradient)
                    nonzero = True
            if not nonzero:
                self.assertEqual(np.sum(gradient), 0)

        self.assertNotEqual(nonzero_sum, 0)

    def test_network_shares_weights(self):
        name = 'gridnet_shared'
        num_input_channels = 3
        num_channels = [num_input_channels, 4, 8]
        grid_width = 2
        num_lateral_convs_per_connection = 1
        num_downsampling_convs_per_connection = 3
        num_upsampling_convs_per_connection = 3
        gridnet = GridNet(num_channels,
                          grid_width,
                          name=name,
                          connection_dropout_rate=1.0,
                          num_lateral_convs=num_lateral_convs_per_connection,
                          num_downsampling_convs=num_downsampling_convs_per_connection,
                          num_upsample_convs=num_upsampling_convs_per_connection,
                          regularizer=l2_regularizer(1e-4))

        height = 32
        width = 60
        num_features = num_input_channels

        # Create the graph.
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        trainable_vars_before = len(tf.trainable_variables())
        gridnet.get_forward(input_features_tensor, training=True, reuse_variables=tf.AUTO_REUSE)
        trainable_vars_after = len(tf.trainable_variables())
        self.assertGreater(trainable_vars_after, trainable_vars_before)

        # Do it again and check that the number of trainable variables has not increased.
        gridnet.get_forward(input_features_tensor, training=True, reuse_variables=tf.AUTO_REUSE)
        self.assertEqual(trainable_vars_after, len(tf.trainable_variables()))
