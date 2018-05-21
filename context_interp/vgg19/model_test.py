import numpy as np
import tensorflow as tf
import unittest
import inspect
import os
from context_interp.vgg19.model import Vgg19


class TestVgg19(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        name = 'vgg19_test'
        self.sess = tf.Session(config=config)
        self.vgg19 = Vgg19(name=name, data_dict=Vgg19.load_data_dict(load_small=True))

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 32
        width = 32
        num_features = 3
        batch_size = 3

        # Create the graph.
        input_image = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        features, layers = self.vgg19.build_up_to_conv4_4(input_image, trainable=False)

        input_images_np = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_images_np[:, 2:height-2, 2:width-2, :] = 3.0

        self.sess.run(tf.global_variables_initializer())
        features = self.sess.run(features, feed_dict={input_image: input_images_np})

        # Test that the default values are working.
        self.assertTrue(np.allclose(features.shape, np.asarray([batch_size, height / 8, width / 8, 512])))
        self.assertNotEqual(np.sum(features), 0.0)

        # Model should not be trainable.
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19')
        trainable_vars = tf.trainable_variables(scope='vgg19')
        self.assertEqual(len(vars), 24)
        self.assertEqual(len(trainable_vars), 0)
        tf.reset_default_graph()


if __name__ == '__main__':
    unittest.main()