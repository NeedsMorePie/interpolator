import tensorflow as tf
from context_interp.vgg19.vgg19 import Vgg19


class ContextExtractor:
    def __init__(self, vgg19=None, name='context_extractor'):
        """
        :param vgg19: An instance of Vgg19. Provide this to prevent re-loading of weights.
        :param name: Str. For variable scoping.
        """
        if vgg19 is None:
            self.vgg19 = Vgg19()
        else:
            self.vgg19 = vgg19

        self.name = name

    def get_forward(self, images):
        """
        :param images: Tensor. Images of shape [batch_size, H, W, 3].
        :return: Tensor. Feature map of shape [batch_size, H, W, num_features].
        """
        with tf.variable_scope(self.name):
            self.vgg19.build_to_conv1_2(images, trainable=False)
            return self.vgg19.conv1_2
