import numpy as np
import tensorflow as tf
import unittest
import inspect
import os
from context_interp.feature_extractors.vgg19.vgg19_features import Vgg19Features
from utils.img import read_image, show_image

VISUALIZE = False


class TestVgg19Features(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.feature_extractor = Vgg19Features()

    def test_features(self):
        height = 32
        width = 32
        num_features = 3
        batch_size = 3

        # Create the graph.
        input_image = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        context_feature_tensor = self.feature_extractor.get_context_features(input_image)
        perceptual_feature_tensor = self.feature_extractor.get_perceptual_features(input_image)

        input_images_np = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        input_images_np[:, 2:height-2, 2:width-2, :] = 3.0

        self.sess.run(tf.global_variables_initializer())
        query = [context_feature_tensor, perceptual_feature_tensor]
        context_features, perceptual_features = self.sess.run(query, feed_dict={input_image: input_images_np})

        # Test that the default values are working.
        self.assertTrue(np.allclose(perceptual_features.shape, np.asarray([batch_size, height / 8, width / 8, 512])))
        self.assertTrue(np.allclose(context_features.shape, np.asarray([batch_size, height, width, 64])))
        self.assertNotEqual(np.sum(context_features), 0.0)
        self.assertNotEqual(np.sum(perceptual_features), 0.0)

        # Models should not be trainable.
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19')
        trainable_vars = tf.trainable_variables(scope='vgg19')
        self.assertEqual(len(vars), 24 + 4)
        self.assertEqual(len(trainable_vars), 0)

        tf.reset_default_graph()

    def test_visualization(self):
        if not VISUALIZE:
            return

        # Load image.
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        img_path = os.path.join(cur_dir, '..', 'test_data', 'hamid.jpg')
        img = read_image(img_path, as_float=True)

        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        img = [img]

        # Create the graph.
        input_image = tf.placeholder(shape=[None, height, width, channels], dtype=tf.float32)
        context_feature_tensor = self.feature_extractor.get_context_features(input_image)
        perceptual_feature_tensor = self.feature_extractor.get_perceptual_features(input_image)

        self.sess.run(tf.global_variables_initializer())
        query = [context_feature_tensor, perceptual_feature_tensor]
        context_features, perceptual_features = self.sess.run(query, feed_dict={input_image: img})

        # Visualize context features.
        for i in range(0, 32, 4):
            show_image(context_features[0][..., i])

        # Visualize perceptual features.
        for i in range(0, 512, 64):
            show_image(perceptual_features[0][..., i])


if __name__ == '__main__':
    unittest.main()