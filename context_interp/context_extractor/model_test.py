import numpy as np
import tensorflow as tf
import unittest
from context_interp.context_extractor.model import ContextExtractor


class TestContextExtractor(unittest.TestCase):
    def setUp(self):
        self.context_extractor = ContextExtractor()

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
        results = self.sess.run(features, feed_dict={input_image: input_images_np})

        # Test that the default values are working.
        self.assertTrue(np.allclose(results[1].shape, np.asarray([batch_size, height/2, width/2, 16])))

        for i in range(1, 13):
            self.assertNotEqual(np.sum(results[i]), 0.0)

        # Test that we have all the trainable variables.
        trainable_vars = tf.trainable_variables(scope='feature_pyramid_network')
        self.assertEqual(len(trainable_vars), 24)

        # Check that the gradients are flowing.
        grad_op = tf.gradients(tf.reduce_mean(final_features), trainable_vars + [input_image])
        gradients = self.sess.run(grad_op, feed_dict={input_image: input_features})
        for gradient in gradients:
            self.assertNotAlmostEqual(np.sum(gradient), 0.0)

        c_3_tensor = layer_outputs[self.feature_pyr_net.get_c_n(3)]
        c_3 = self.sess.run(c_3_tensor, feed_dict={input_image: input_features})
        self.assertTrue(np.allclose(c_3.shape, [batch_size, 64, 64, 64]))

        c_5_tensor = layer_outputs[self.feature_pyr_net.get_c_n(5)]
        c_5 = self.sess.run(c_5_tensor, feed_dict={input_image: input_features})
        self.assertTrue(np.allclose(c_5.shape, [batch_size, 16, 16, 128]))


if __name__ == '__main__':
    unittest.main()