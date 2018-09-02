import numpy as np
import os
import os.path
import tensorflow as tf
import unittest
from common.models import ConvNetwork, RestorableNetwork


class TestRestorableModel(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.output_path = os.path.join('common', 'test_output.npz')
        self.maxDiff = None

    def test_save_restore_np(self):
        """
        Basic save and restore test.
        """
        layer_specs = [[3, 7, 1, 1],
                       [3, 6, 1, 1]]
        conv_net = ConvNetwork('conv_network_test_save_restore', layer_specs=layer_specs)

        # Create and initialize model.
        input_placeholder = tf.placeholder(shape=[None, 16, 16, 2], dtype=tf.float32)
        output, _, _ = conv_net.get_forward_conv(input_placeholder)
        global_initializer = tf.global_variables_initializer()
        self.sess.run(global_initializer)

        # Get the output.
        dummy_input = np.ones(shape=[1, 16, 16, 2], dtype=np.float32)
        dummy_output = self.sess.run(output, feed_dict={input_placeholder: dummy_input})

        # Save and test that the expected variables were saved.
        saved_np = conv_net.get_save_np(sess=self.sess)
        sorted_keys = sorted(saved_np.keys())
        expected_keys = ['conv_network_test_save_restore/conv_0/bias:0',
                         'conv_network_test_save_restore/conv_0/kernel:0',
                         'conv_network_test_save_restore/conv_1/bias:0',
                         'conv_network_test_save_restore/conv_1/kernel:0']
        self.assertListEqual(expected_keys, sorted_keys)

        # Run the initializer again and make sure the output is different.
        self.sess.run(global_initializer)
        dummy_output_not_same = self.sess.run(output, feed_dict={input_placeholder: dummy_input})
        self.assertFalse(np.allclose(dummy_output, dummy_output_not_same))

        # Restore and run again.
        conv_net.restore_from_np(saved_np, self.sess)
        same_dummy_output = self.sess.run(output, feed_dict={input_placeholder: dummy_input})
        self.assertTrue(np.allclose(dummy_output, same_dummy_output))

    def test_save_restore_file(self):
        """
        Basic save and restore test.
        """
        layer_specs = [[3, 7, 1, 1],
                       [3, 6, 1, 1],
                       [3, 5, 1, 1]]
        conv_net = ConvNetwork('conv_network_file_test', layer_specs=layer_specs)

        # Create and initialize model.
        input_placeholder = tf.placeholder(shape=[None, 4, 4, 2], dtype=tf.float32)
        output, _, _ = conv_net.get_forward_conv(input_placeholder)
        global_initializer = tf.global_variables_initializer()
        self.sess.run(global_initializer)

        # Get the output.
        dummy_input = np.ones(shape=[1, 4, 4, 2], dtype=np.float32)
        dummy_output = self.sess.run(output, feed_dict={input_placeholder: dummy_input})

        # Save and test that the expected variables were saved.
        conv_net.save_to(self.output_path, sess=self.sess)

        # Run the initializer again and make sure the output is different.
        self.sess.run(global_initializer)
        dummy_output_not_same = self.sess.run(output, feed_dict={input_placeholder: dummy_input})
        self.assertFalse(np.allclose(dummy_output, dummy_output_not_same))

        # Restore and run again.
        conv_net.restore_from(self.output_path, self.sess)
        same_dummy_output = self.sess.run(output, feed_dict={input_placeholder: dummy_input})
        self.assertTrue(np.allclose(dummy_output, same_dummy_output))

    def test_network_size_change(self):
        """
        Tests that we can transfer weights from one network to another that has a potentially different size.
        """
        layer_specs = [[3, 7, 1, 1],
                       [3, 6, 1, 1]]
        conv_net_small = ConvNetwork('conv_network_small_test', layer_specs=layer_specs)
        conv_net_large = ConvNetwork('conv_network_large_test', layer_specs=layer_specs)

        # Create and initialize models.
        input_small = tf.placeholder(shape=[None, 8, 8, 2], dtype=tf.float32)
        output_small, _, _ = conv_net_small.get_forward_conv(input_small)
        input_large = tf.placeholder(shape=[None, 16, 16, 2], dtype=tf.float32)
        output_large, _, _ = conv_net_large.get_forward_conv(input_large)
        global_initializer = tf.global_variables_initializer()
        self.sess.run(global_initializer)

        # Run and check that they aren't the same.
        dummy_input_small = np.ones(shape=[1, 8, 8, 2], dtype=np.float32)
        dummy_output_small = self.sess.run(output_small, feed_dict={input_small: dummy_input_small})
        dummy_input_large = np.ones(shape=[1, 16, 16, 2], dtype=np.float32)
        dummy_output_large = self.sess.run(output_large, feed_dict={input_large: dummy_input_large})
        self.assertFalse(np.allclose(dummy_output_small[0, 4, 4, :], dummy_output_large[0, 8, 8, :]))

        # Convert the small network's dict to the large network's dict.
        large_var_dict = RestorableNetwork.rename_np_dict(conv_net_small.get_save_np(self.sess),
                                                          conv_net_small.name, conv_net_large.name)

        # Restore the small network's weights into the large network.
        conv_net_large.restore_from_np(large_var_dict, self.sess)

        # Run the large network again and the output should match now.
        dummy_output_large = self.sess.run(output_large, feed_dict={input_large: dummy_input_large})
        self.assertTrue(np.allclose(dummy_output_small[0, 4, 4, :], dummy_output_large[0, 8, 8, :]))

    def test_network_name_and_scope_change(self):
        """
        Tests that we can transfer weights from one network to another that has different name and a deeper scope.
        """
        layer_specs = [[3, 7, 1, 1],
                       [3, 6, 1, 1]]
        scope_name = 'other_scope'

        # Create and initialize models.
        conv_net_small = ConvNetwork('conv_network_small_test', layer_specs=layer_specs)
        input_small = tf.placeholder(shape=[None, 8, 8, 2], dtype=tf.float32)
        output_small, _, _ = conv_net_small.get_forward_conv(input_small)
        with tf.variable_scope(scope_name):
            conv_net_large = ConvNetwork('conv_network_large_test', layer_specs=layer_specs)
            input_large = tf.placeholder(shape=[None, 16, 16, 2], dtype=tf.float32)
            output_large, _, _ = conv_net_large.get_forward_conv(input_large)
        global_initializer = tf.global_variables_initializer()
        self.sess.run(global_initializer)

        # Run and check that they aren't the same.
        dummy_input_small = np.ones(shape=[1, 8, 8, 2], dtype=np.float32)
        dummy_output_small = self.sess.run(output_small, feed_dict={input_small: dummy_input_small})
        dummy_input_large = np.ones(shape=[1, 16, 16, 2], dtype=np.float32)
        dummy_output_large = self.sess.run(output_large, feed_dict={input_large: dummy_input_large})
        self.assertFalse(np.allclose(dummy_output_small[0, 4, 4, :], dummy_output_large[0, 8, 8, :]))

        # Convert the small network's dict to the large network's dict.
        large_var_dict = RestorableNetwork.rename_np_dict(conv_net_small.get_save_np(self.sess),
                                                          conv_net_small.name, conv_net_large.name)

        # Restore the small network's weights into the large network.
        conv_net_large.restore_from_np(large_var_dict, self.sess, scope_prefix=scope_name)

        # Run the large network again and the output should match now.
        dummy_output_large = self.sess.run(output_large, feed_dict={input_large: dummy_input_large})
        self.assertTrue(np.allclose(dummy_output_small[0, 4, 4, :], dummy_output_large[0, 8, 8, :]))

    def tearDown(self):
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)


class TestConvNet(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_default_not_dense(self):
        input_features = 2
        layer_1_features = 7
        layer_2_features = 6
        image_size = 8
        batch_size = 2

        layer_specs = [[3, layer_1_features, 1, 1],
                       [3, layer_2_features, 1, 1]]
        conv_net = ConvNetwork('conv_network_test_not_dense', layer_specs=layer_specs)

        # Create and initialize model.
        input_placeholder = tf.placeholder(shape=[None, image_size, image_size, 2], dtype=tf.float32)
        output, layer_outputs, dense_outputs = conv_net.get_forward_conv(input_placeholder)
        self.assertEqual(0, len(dense_outputs))
        self.assertEqual(2, len(layer_outputs))
        self.assertEqual(output, layer_outputs[-1])

        self.sess.run(tf.global_variables_initializer())
        dummy_input = np.ones(shape=[batch_size, image_size, image_size, input_features], dtype=np.float32)
        dummy_output, dummy_outputs = self.sess.run([output, layer_outputs], feed_dict={input_placeholder: dummy_input})

        self.assertTupleEqual(dummy_outputs[0].shape, (batch_size, image_size, image_size, layer_1_features))
        self.assertTupleEqual(dummy_output.shape, (batch_size, image_size, image_size, layer_2_features))

    def test_dense(self):
        input_features = 2
        layer_1_features = 7
        layer_2_features = 6
        image_size = 8

        layer_specs = [[3, layer_1_features, 1, 1],
                       [3, layer_2_features, 1, 1]]
        conv_net = ConvNetwork('conv_network_test_dense', layer_specs=layer_specs, dense_net=True)

        # Create and initialize model.
        input_placeholder = tf.placeholder(shape=[None, image_size, image_size, input_features], dtype=tf.float32)
        output, layer_outputs, dense_outputs = conv_net.get_forward_conv(input_placeholder)
        self.assertEqual(2, len(dense_outputs))
        self.assertEqual(output, layer_outputs[-1])

        self.sess.run(tf.global_variables_initializer())
        dummy_input = np.ones(shape=[1, image_size, image_size, input_features], dtype=np.float32)
        dense_input_dummy, dummy_outputs = self.sess.run([dense_outputs, layer_outputs],
                                                         feed_dict={input_placeholder: dummy_input})

        self.assertTupleEqual(dense_input_dummy[0].shape,
                              (1, image_size, image_size, input_features + layer_1_features))
        self.assertTupleEqual(dense_input_dummy[1].shape,
                              (1, image_size, image_size, input_features + layer_1_features + layer_2_features))

        self.assertTupleEqual(dummy_outputs[0].shape, (1, image_size, image_size, layer_1_features))
        self.assertTupleEqual(dummy_outputs[1].shape, (1, image_size, image_size, layer_2_features))

    def test_share_variables(self):
        input_features = 2
        layer_1_features = 7
        layer_2_features = 6
        image_size = 8

        layer_specs = [[3, layer_1_features, 1, 1],
                       [3, layer_2_features, 1, 1]]
        conv_net = ConvNetwork('conv_network_shared', layer_specs=layer_specs, dense_net=True)

        # Create and initialize model.
        input_placeholder = tf.placeholder(shape=[None, image_size, image_size, input_features], dtype=tf.float32)
        trainable_vars_before = len(tf.trainable_variables())  # Global scope used to make sure we don't miss anything.
        _ = conv_net.get_forward_conv(input_placeholder)
        trainable_vars_after = len(tf.trainable_variables())
        self.assertEqual(4, trainable_vars_after - trainable_vars_before)

        # Do it again and check that the number of trainable variables has not increased.
        _ = conv_net.get_forward_conv(input_placeholder)
        self.assertEqual(trainable_vars_after, len(tf.trainable_variables()))


if __name__ == '__main__':
    unittest.main()
