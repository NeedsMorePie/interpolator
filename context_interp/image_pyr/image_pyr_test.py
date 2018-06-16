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
        kernel_tensor = pyr._get_blur_kernel()
        kernel = self.sess.run(kernel_tensor)
        expected = np.array([
            [1, 3, 3, 1],
            [3, 9, 9, 3],
            [3, 9, 9, 3],
            [1, 3, 3, 1]
        ])
        expected = expected / np.sum(expected)
        self.assertEquals(kernel.tolist(), expected.tolist())

