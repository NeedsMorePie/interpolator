import numpy as np
import tensorflow as tf
import unittest
from context_interp.context_extractor.model import ContextExtractor


class TestContextExtractor(unittest.TestCase):
    def setUp(self):
        self.context_extractor = ContextExtractor(name='context_extractor')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 28
        width = 28
        num_features = 3
        batch_size = 3

        # Create the graph.
        input_image = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        features = self.context_extractor.get_forward(input_image)

        input_images_np = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_images_np[:, 2:height-2, 2:width-2, :] = 3.0

        self.sess.run(tf.global_variables_initializer())
        features = self.sess.run(features, feed_dict={input_image: input_images_np})

        # Test that the default values are working.
        self.assertTrue(np.allclose(features.shape, np.asarray([batch_size, height, width, 64])))
        self.assertNotEqual(np.sum(features), 0.0)

        # Model should not be trainable.
        trainable_vars = tf.trainable_variables(scope='context_extractor')
        self.assertEqual(len(trainable_vars), 0)


if __name__ == '__main__':
    unittest.main()