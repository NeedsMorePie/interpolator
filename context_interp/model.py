import tensorflow as tf
import numpy as np
import os
import inspect
from context_interp.feature_extractors.vgg19.vgg19_features import Vgg19Features
from context_interp.gridnet.model import GridNet
from pwcnet.warp.warp import warp_via_flow
from pwcnet.model import PWCNet

class ContextInterp:
    def __init__(self, name='context_interp'):
        # @TODO Think of what arguments to have ...

        self.name = name
        self.pwcnet = PWCNet()
        self.gridnet = GridNet([32, 64, 96], 6, num_output_channels=3)
        self.feature_extractor = Vgg19Features()

    def get_forward(self, image_a, image_b, t, reuse_variables=tf.AUTO_REUSE):
        """
        :param image_a: Tensor of shape [batch_size, H, W, 3].
        :param image_b: Tensor of shape [batch_size, H, W, 3].
        :param t: Float. Specifies the interpolation point (i.e 0 for image_a, 1 for image_b).
        :return: interpolated: The interpolated image. Tensor of shape [batch_size, H, W, 3].
                 warped_a_b: Image and features from a forward-flowed towards b, before synthesis.
                             The first 3 channels are the image.
                 warped_b_a: Image and features from b forward-flowed towards a, before synthesis.
                             The first 3 channels are the image.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            image_a_contexts = self.feature_extractor.get_context_features(image_a)
            image_b_contexts = self.feature_extractor.get_context_features(image_b)

            # Get a->b and b->a flows from PWCNet.
            # flow_a_b, _ = self.pwcnet.get_forward(image_a, image_b)
            # flow_b_a, _ = self.pwcnet.get_forward(image_b, image_a)
            # flow_a_b = tf.stop_gradient(flow_a_b)
            # flow_b_a = tf.stop_gradient(flow_b_a)

            features_a = tf.concat([image_a, image_a_contexts], axis=-1)
            features_b = tf.concat([image_b, image_b_contexts], axis=-1)

            # Warp images and their contexts from a->b and from b->a.
            # @TODO Forward warp needs to be added. Currently no warp is applied.
            warped_a_b = features_a
            warped_b_a = features_b

            # Feed into GridNet for final synthesis.
            warped_combined = tf.concat([warped_a_b, warped_b_a], axis=-1)
            synthesized, _, _, _ = self.gridnet.get_forward(warped_combined, training=True)
            return synthesized, warped_a_b, warped_b_a

    def get_training_loss(self, prediction, expected):
        """
        :param prediction: Tensor of shape [batch, H, W, num_features]. Predicted image.
        :param expected: Tensor of shape [batch, H, W, num_features]. Ground truth image.
        :return: Tf scalar loss term.
        """
        return self._get_feature_loss(prediction, expected)

    def _get_feature_loss(self, prediction, expected):
        prediction_features = self.feature_extractor.get_perceptual_features(prediction)
        expected_features = self.feature_extractor.get_perceptual_features(expected)
        return tf.reduce_sum(tf.squared_difference(prediction_features, expected_features))

    def _get_l1_loss(self, prediction, expected):
        return tf.reduce_sum(tf.abs(prediction - expected))

    def _get_laplacian_loss(self, prediction):
        raise NotImplementedError

