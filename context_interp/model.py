import tensorflow as tf
from context_interp.vgg19_features.vgg19_features import Vgg19Features
from context_interp.gridnet.model import GridNet
from context_interp.laplacian_pyramid.laplacian_pyramid import LaplacianPyramid
from pwcnet.model import PWCNet
from common.forward_warp.forward_warp import forward_warp


class ContextInterp:
    def __init__(self, name='context_interp'):
        """
        :param name: Str. For Tf variable scoping.
        """
        self.name = name
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
        with tf.variable_scope(self.name, reuse=reuse_variables):
            image_a_contexts = self.feature_extractor.get_context_features(image_a)
            image_b_contexts = self.feature_extractor.get_context_features(image_b)

            # Get a->b and b->a flows from PWCNet.
            flow_a_b, _ = self.pwcnet.get_forward(image_a, image_b)
            flow_b_a, _ = self.pwcnet.get_forward(image_b, image_a)

            features_a = tf.concat([image_a, image_a_contexts], axis=-1)
            features_b = tf.concat([image_b, image_b_contexts], axis=-1)

            # Warp images and their contexts from a->b and from b->a.
            warped_a_b = forward_warp(features_a, t * flow_a_b)
            warped_b_a = forward_warp(features_b, (1.0 - t) * flow_b_a)

            # Feed into GridNet for final synthesis.
            warped_combined = tf.concat([warped_a_b, warped_b_a], axis=-1)
            warped_combined = tf.stop_gradient(warped_combined)
            synthesized, _, _, _ = self.gridnet.get_forward(warped_combined, training=True)
            return synthesized, warped_a_b, warped_b_a, flow_a_b, flow_b_a

    def load_pwcnet_weights(self, pwcnet_weights_path, sess):
        """
        Loads pre-trained PWCNet weights. Must be called after get_forward.
        :param pwcnet_weights_path: The full path to PWCNet weights that will be loaded via
                                    the RestorableNetwork interface.
        :param sess: Tf Session.
        """
        self.pwcnet.restore_from(pwcnet_weights_path, sess)

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
        with tf.variable_scope('laplacian_loss'):
            pyr1, _, _ = self.laplacian_pyramid.get_forward(prediction)
            pyr2, _, _ = self.laplacian_pyramid.get_forward(expected)
            loss = 0
            for i in range(len(pyr2)):
                loss += 2 ** i * tf.reduce_sum(tf.abs(pyr1[i] - pyr2[i]))
            return loss
