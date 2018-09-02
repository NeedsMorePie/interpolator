import numpy as np
import tensorflow as tf
import unittest
from context_interp.gridnet.connections.connections import LateralConnection, UpSamplingConnection, DownSamplingConnection
from tensorflow.contrib.layers import l2_regularizer


class TestConnections(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_lateral(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 30
        width = 16
        num_features = 32
        batch_size = 3
        num_output_features = 32  # Must be the same as num_features as it has a skip connection

        specs = [
            [8, 1],
            [16, 1],
            [num_output_features, 1]
        ]

        # Create the graph.
        name = 'lateral_connection'
        connection = LateralConnection(name, specs, regularizer=l2_regularizer(1e-4))
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        output = connection.get_forward(input_features_tensor)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 2:height-2, 1:width-1, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [output]
        results = self.sess.run(query, feed_dict={input_features_tensor: input_features})

        self.assertEqual(len(results), 1)
        output_np = results[0]
        self.assertTrue(np.allclose(output_np.shape, np.asarray([batch_size, height, width, num_output_features])))
        self.assertNotEqual(np.sum(output_np), 0.0)

        # Test regularization losses.
        # len(specs) conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope=name)
        self.assertEqual(len(reg_losses), 2 * len(specs))

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        self.assertEqual(reg_losses[0].name, name + '/conv_0/kernel/Regularizer/l2_regularizer:0')

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope=name)
        self.assertEqual(len(trainable_vars), 2 * len(specs))
        self.assertEqual(trainable_vars[2].name, name + '/conv_1/kernel:0')

        # Test that getting forward again with tf.AUTO_REUSE will not increase the number of variables.
        num_trainable_vars = len(tf.trainable_variables())
        connection.get_forward(input_features_tensor, reuse_variables=tf.AUTO_REUSE)
        self.assertEqual(num_trainable_vars, len(tf.trainable_variables()))

    def test_downsampling(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 32
        width = 30
        num_features = 16
        batch_size = 3
        num_output_features = 8

        specs = [
            [8, 1],
            [64, 1],
            [32, 1],
            [num_output_features, 1]
        ]

        # Create the graph.
        name = 'downsampling_connection'
        connection = DownSamplingConnection(name, specs, regularizer=l2_regularizer(1e-4))
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        output = connection.get_forward(input_features_tensor)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 0:height - 4, 0:width - 3, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [output]
        results = self.sess.run(query, feed_dict={input_features_tensor: input_features})

        self.assertEqual(len(results), 1)
        output_np = results[0]
        self.assertTrue(np.allclose(output_np.shape, np.asarray([batch_size, height / 2, width / 2, num_output_features])))
        self.assertNotEqual(np.sum(output_np), 0.0)

        # Test regularization losses.
        # len(specs) conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope=name)
        self.assertEqual(len(reg_losses), 2 * len(specs))

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        self.assertEqual(reg_losses[0].name, name + '/conv_0/kernel/Regularizer/l2_regularizer:0')

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope=name)
        self.assertEqual(len(trainable_vars), 2 * len(specs))
        self.assertEqual(trainable_vars[2].name, name + '/conv_1/kernel:0')

        # Test that getting forward again with tf.AUTO_REUSE will not increase the number of variables.
        num_trainable_vars = len(tf.trainable_variables())
        connection.get_forward(input_features_tensor, reuse_variables=tf.AUTO_REUSE)
        self.assertEqual(num_trainable_vars, len(tf.trainable_variables()))

    def test_upsampling(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 8
        width = 4
        num_features = 32
        batch_size = 3
        num_output_features = 64

        specs = [
            [num_output_features, 1]
        ]

        # Create the graph.
        name = 'upsampling_connection'
        connection = UpSamplingConnection(name, specs,
                                          regularizer=l2_regularizer(1e-4))
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        output = connection.get_forward(input_features_tensor)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 0:height, 0:width, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [output]
        results = self.sess.run(query, feed_dict={input_features_tensor: input_features})

        self.assertEqual(len(results), 1)
        output_np = results[0]
        self.assertTrue(np.allclose(output_np.shape, np.asarray([batch_size, height * 2, width * 2, num_output_features])))
        self.assertNotEqual(np.sum(output_np), 0.0)

        # Test regularization losses.
        # len(specs) conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope=name)
        self.assertEqual(len(reg_losses), 2 * len(specs))

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        self.assertEqual(reg_losses[0].name, name + '/conv_0/kernel/Regularizer/l2_regularizer:0')

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope=name)
        self.assertEqual(len(trainable_vars), 2 * len(specs))
        self.assertEqual(trainable_vars[0].name, name + '/conv_0/kernel:0')

        # Test that getting forward again with tf.AUTO_REUSE will not increase the number of variables.
        num_trainable_vars = len(tf.trainable_variables())
        connection.get_forward(input_features_tensor, reuse_variables=tf.AUTO_REUSE)
        self.assertEqual(num_trainable_vars, len(tf.trainable_variables()))

    def test_lateral_dropout(self):
        """
        Makes sure that total dropout is working.
        """
        height = 8
        width = 4
        num_features = 64
        batch_size = 3
        num_output_features = 64  # Must be the same as num_features as it has a skip connection

        specs = [
            [num_output_features, 1]
        ]

        # Create the graph.
        name = 'lateral_connection_dropped'
        connection = LateralConnection(name, specs,
                                          regularizer=l2_regularizer(1e-4),
                                          total_dropout_rate=1.0)
        input_features_tensor = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        output = connection.get_forward(input_features_tensor, training=True)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 0:height, 0:width, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [output]
        results = self.sess.run(query, feed_dict={input_features_tensor: input_features})

        self.assertEqual(len(results), 1)
        output_np = results[0]
        self.assertTrue(np.allclose(output_np.shape, np.asarray([batch_size, height, width, num_output_features])))

        # As we applied total dropout, the output of the connection should be 0.
        self.assertEqual(np.sum(output_np), 0.0)

        # Test regularization losses.
        # len(specs) conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope=name)
        self.assertEqual(len(reg_losses), 2 * len(specs))

        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        self.assertEqual(reg_losses[0].name, name + '/conv_0/kernel/Regularizer/l2_regularizer:0')

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope=name)
        self.assertEqual(len(trainable_vars), 2 * len(specs))
        self.assertEqual(trainable_vars[0].name, name + '/conv_0/kernel:0')

if __name__ == '__main__':
    unittest.main()
