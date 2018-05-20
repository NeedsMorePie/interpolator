import tensorflow as tf
import inspect
import os
from context_interp.vgg19.vgg19 import Vgg19


class ContextExtractor:
    def __init__(self, vgg19=None, name='context_extractor'):
        """
        :param vgg19: An instance of Vgg19. Provide this to prevent re-loading of weights.
        :param name: Str. For variable scoping.
        """
        if vgg19 is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19_conv4_4.npy")
            vgg19_npy_path = path
            self.vgg19 = Vgg19(vgg19_npy_path=vgg19_npy_path)
        else:
            self.vgg19 = vgg19

        self.name = name

    def get_forward(self, images):
        """
        :param images: Tensor. Images of shape [batch_size, H, W, 3].
        :return: Tensor. Feature map of shape [batch_size, H, W, num_features].
        """
        with tf.variable_scope(self.name):
            self.vgg19.build_up_to_conv1_2(images, trainable=False)
            return self.vgg19.conv1_2
