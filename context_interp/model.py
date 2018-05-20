import tensorflow as tf
import numpy as np
import os
import inspect
from context_interp.vgg19.model import Vgg19
from pwcnet.warp.warp import warp_via_flow
from pwcnet.model import PWCNet

class ContextInterp:
    def __init__(self, name='context_interp'):
        # @TODO Think of what arguments to have ...

        # Load VGGNet constants (and not all of them).
        vgg_path = inspect.getfile(Vgg19)
        vgg_path = os.path.abspath(os.path.join(vgg_path, os.pardir))
        vgg_path = os.path.join(vgg_path, "vgg19_conv4_4.npy")
        self.vgg19_data = Vgg19.load_data_dict(vgg19_npy_path=vgg_path)
        self.vgg19 = Vgg19(self.vgg19_data)

    def get_forward(self, image_a, image_b, t):
        """
        :param image_a: Tensor of shape [batch_size, H, W, 3].
        :param image_b: Tensor of shape [batch_size, H, W, 3].
        :param t: Float. Must fall in range [0, 1] inclusive.
        :return: interpolated: The interpolated image. Tensor of shape [batch_size, H, W, 3].
                 warped_image_a_b: Image a forward-flowed towards b, before synthesis.
                 warped_image_b_a: Image b forward-flowed towards a, before synthesis.
                 warped_context_a_b: Features a forward-flowed towards b, before synthesis.
                                     Tensor of shape [batch_size, H, W, 32].
                 warped_context_b_a: Features b forward-flowed towards a, before synthesis.
                                     Tensor of shape [batch_size, H, W, 32].
        """
        image_a_contexts, _ = self.vgg19.get_forward_up_to_conv1_2(image_a, trainable=False)
        image_b_contexts, _ = self.vgg19.get_forward_up_to_conv1_2(image_b, trainable=False)

        # Get a to b and b to a flows from PWCNet.

        # Warp images and their contexts from a to b and from b to a.

        # Stack and feed into GridNet.


    def get_feature_loss(self, prediction, expected):
        """
        Uses VGG19 layer conv4_4 to compute squared distance loss.
        :param prediction: Predicted image.
        :param expected: Ground truth image.
        :return: Tf scalar loss term.
        """
        prediction_features = self.vgg19.get_forward_up_to_conv4_4(prediction, trainable=False)
        expected_features = self.vgg19.get_forward_up_to_conv4_4(expected, trainable=False)
        return tf.reduce_sum(tf.squared_difference(prediction_features, expected_features))
