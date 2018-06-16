import numpy as np
import tensorflow as tf
import unittest
from context_interp.image_pyr.image_pyr import ImagePyramid


class TestImagePyramid(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def testKernel(self):
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

