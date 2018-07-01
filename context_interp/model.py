import tensorflow as tf
import numpy as np
from context_interp.vgg19_features.vgg19_features import Vgg19Features
from context_interp.gridnet.model import GridNet
from context_interp.laplacian_pyramid.laplacian_pyramid import LaplacianPyramid
from pwcnet.model import PWCNet
from common.models import RestorableNetwork
from common.forward_warp.forward_warp import forward_warp


class ContextInterp:
    def __init__(self, name='context_interp'):
        """
        :param name: Str. For Tf variable scoping.
        """
        self.name = name
        self.enclosing_scope = None
        self.gridnet = GridNet([32, 64, 96], 6, num_output_channels=3)
        self.laplacian_pyramid = LaplacianPyramid(5)
        self.pwcnet = PWCNet()
        self.feature_extractor = Vgg19Features()
        self.feature_extractor.load_pretrained_weights()

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
                 flow_a_b: Flow from a to b (centered at a).
                 flow_b_a: Flow from b to a (centered at b).
        """
        self.enclosing_scope = tf.get_variable_scope()
        with tf.variable_scope(self.name, reuse=reuse_variables):
            image_a_contexts = self.feature_extractor.get_context_features(image_a)
            image_b_contexts = self.feature_extractor.get_context_features(image_b)

            # TODO: Add instance normalization. Described in 3.3 of https://arxiv.org/pdf/1803.10967.pdf.

            # Get a->b and b->a flows from PWCNet.
            flow_a_b, _ = self.pwcnet.get_forward(image_a, image_b)
            flow_b_a, _ = self.pwcnet.get_forward(image_b, image_a)

            features_a = tf.concat([image_a, image_a_contexts], axis=-1)
            features_b = tf.concat([image_b, image_b_contexts], axis=-1)

            # Warp images and their contexts from a->b and from b->a.
            warped_a_b = forward_warp(features_a, t * flow_a_b)
            warped_b_a = forward_warp(features_b, (1.0 - t) * flow_b_a)
            warped_a_b = tf.stop_gradient(warped_a_b)
            warped_b_a = tf.stop_gradient(warped_b_a)

            # Feed into GridNet for final synthesis.
            warped_combined = tf.concat([warped_a_b, warped_b_a], axis=-1)
            synthesized, _, _, _ = self.gridnet.get_forward(warped_combined, training=True)
            return synthesized, warped_a_b, warped_b_a, flow_a_b, flow_b_a

    def load_pwcnet_weights(self, pwcnet_weights_path, sess):
        """
        Loads pre-trained PWCNet weights.
        For this to work:
            - It must be called after get_forward.
            - The pwcnet weights must have been saved under variable scope 'pwcnet'.
        :param pwcnet_weights_path: The full path to the PWCNet weights that will be loaded via
                                    the RestorableNetwork interface.
        :param sess: Tf Session.
        """
        assert self.enclosing_scope is not None, 'get_forward must have been called beforehand.'
        scope_prefix = self.enclosing_scope.name + self.name
        self.pwcnet.restore_from(pwcnet_weights_path, sess, scope_prefix=scope_prefix)

    def get_training_loss(self, prediction, expected):
        """
        :param prediction: Tensor of shape [batch, H, W, num_features]. Predicted image.
        :param expected: Tensor of shape [batch, H, W, num_features]. Ground truth image.
        :return: Tf scalar loss term.
        """
        return self._get_laplacian_loss(prediction, expected)

    def get_fine_tuning_loss(self, prediction, expected):
        """
        :param prediction: Tensor of shape [batch, H, W, num_features]. Predicted image.
        :param expected: Tensor of shape [batch, H, W, num_features]. Ground truth image.
        :return: Tf scalar loss term.
        """
        return self._get_feature_loss(prediction, expected)

    def _get_feature_loss(self, prediction, expected):
        with tf.variable_scope('feature_loss'):
            prediction_features = self.feature_extractor.get_perceptual_features(prediction)
            expected_features = self.feature_extractor.get_perceptual_features(expected)
            return tf.reduce_mean(tf.squared_difference(prediction_features, expected_features))

    def _get_l1_loss(self, prediction, expected):
        return tf.reduce_mean(tf.abs(prediction - expected))

    def _get_laplacian_loss(self, prediction, expected):
        with tf.name_scope('laplacian_loss'):
            pyr1, _, _ = self.laplacian_pyramid.get_forward(prediction)
            pyr2, _, _ = self.laplacian_pyramid.get_forward(expected)
            loss = 0
            for i in range(len(pyr2)):
                loss += 2 ** i * tf.reduce_sum(tf.abs(pyr1[i] - pyr2[i]))
            return loss
