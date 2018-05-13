import numpy as np
import tensorflow as tf
import unittest
from pwcnet.feature_pyramid_network.model import FeaturePyramidNetwork
from tensorflow.contrib.layers import l2_regularizer


class TestFeaturePyramid(unittest.TestCase):
    def setUp(self):
        self.feature_pyr_net = FeaturePyramidNetwork(regularizer=l2_regularizer(1e-4))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 512
        width = 512
        num_features = 3
        batch_size = 3

        # Create the graph.
        input_image = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        final_features, layer_outputs = self.feature_pyr_net.get_forward(input_image)

        input_features = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_features[:, 10:height-10, 10:width-10, :] = 1.0

        self.sess.run(tf.global_variables_initializer())

        query = [final_features] + layer_outputs
        results = self.sess.run(query, feed_dict={input_image: input_features})

        self.assertEqual(len(results), 13)

        # Test that the default values are working.
        self.assertTrue(np.allclose(results[1].shape, np.asarray([batch_size, height/2, width/2, 16])))
        self.assertTrue(np.allclose(results[3].shape, np.asarray([batch_size, height/4, width/4, 32])))
        self.assertTrue(np.allclose(results[5].shape, np.asarray([batch_size, height/8, width/8, 64])))
        self.assertTrue(np.allclose(results[7].shape, np.asarray([batch_size, height/16, width/16, 96])))
        self.assertTrue(np.allclose(results[9].shape, np.asarray([batch_size, height/32, width/32, 128])))
        self.assertTrue(np.allclose(results[11].shape, np.asarray([batch_size, height/64, width/64, 192])))

        for i in range(1, 13):
            self.assertNotEqual(np.sum(results[i]), 0.0)

        # Test regularization losses.
        # 7 conv layers x2 (bias and kernels).
        reg_losses = tf.losses.get_regularization_losses(scope='feature_pyramid_network')
        self.assertEqual(len(reg_losses), 24)
        # Make sure the reg losses aren't 0.
        reg_loss_sum_tensor = tf.add_n(reg_losses)
        reg_loss_sum = self.sess.run(reg_loss_sum_tensor)
        self.assertNotEqual(reg_loss_sum, 0.0)

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope='feature_pyramid_network')
        self.assertEqual(len(trainable_vars), 24)

        # Check that the gradients are flowing.
        grad_op = tf.gradients(tf.reduce_mean(final_features), trainable_vars + [input_image])
        gradients = self.sess.run(grad_op, feed_dict={input_image: input_features})
        for gradient in gradients:
            self.assertNotAlmostEqual(np.sum(gradient), 0.0)


if __name__ == '__main__':
    unittest.main()
