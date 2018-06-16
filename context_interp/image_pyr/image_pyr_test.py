import numpy as np
import tensorflow as tf
import unittest
import os
from utils.img import read_image, show_image
from context_interp.image_pyr.image_pyr import ImagePyramid


VISUALIZE = True

class TestImagePyramid(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Load Lena.
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(cur_dir, 'test_data', 'lena.jpg')
        self.test_image = read_image(image_path)

    def testPyramid(self):
        num_levels = 5
        pyr_builder = ImagePyramid(num_levels, filter_side_len=5)
        image_height = np.shape(self.test_image)[0]
        image_width = np.shape(self.test_image)[1]
        image_tensor = tf.placeholder(tf.float32, shape=(1, image_height, image_width, 3))
        pyr_tensors = pyr_builder.get_forward(image_tensor)
        pyr = self.sess.run(pyr_tensors, feed_dict={image_tensor: [self.test_image]})

        # Check shapes.
        self.assertEqual(len(pyr), num_levels)
        for i, level in enumerate(pyr):
            self.assertTupleEqual(np.shape(level), (1, image_height / 2 ** i, image_width / 2 ** i, 3))

        # Check that gradients flow.
        grads_tensor = tf.gradients(pyr_tensors, image_tensor)
        grads = self.sess.run(grads_tensor, feed_dict={image_tensor: [self.test_image]})
        for grad in grads:
            self.assertNotEqual(np.sum(grad), 0.0)

        if VISUALIZE:
            for level in pyr:
                show_image(level[0] / 255.0)

    def testFilter(self):
        pyr = ImagePyramid(5, filter_side_len=4)
        filter_tensor = pyr._get_blur_filter(3)
        filter = self.sess.run(filter_tensor)
        expected = np.array([
            [1, 3, 3, 1],
            [3, 9, 9, 3],
            [3, 9, 9, 3],
            [1, 3, 3, 1]
        ])
        expected = expected / np.sum(expected)
        self.assertTupleEqual(np.shape(filter), (4, 4, 3, 1))
        self.assertEqual(filter[..., 0, 0].tolist(), expected.tolist())

