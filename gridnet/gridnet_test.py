import numpy as np
import tensorflow as tf
import unittest
from gridnet.gridnet import GridNet


class TestGridNet(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
