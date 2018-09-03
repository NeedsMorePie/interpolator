import tensorflow as tf
from common.forward_warp.forward_warp import forward_warp
from interp.interp import Interp
from interp.context_interp.vgg19_features.vgg19_features import Vgg19Features
from interp.context_interp.gridnet.model import GridNet
from interp.context_interp.laplacian_pyramid.laplacian_pyramid import LaplacianPyramid
from pwcnet.model import PWCNet


class ContextInterp(Interp):
    def __init__(self, name='context_interp', saved_model_dir=None):
        """
        :param name: Str. For Tf variable scoping.
        :param saved_model_dir: See parent class.
        """
        super().__init__(name, saved_model_dir=saved_model_dir)
        self.enclosing_scope = None
        self.gridnet = GridNet([32, 64, 96], 6, num_output_channels=3)
        self.laplacian_pyramid = LaplacianPyramid(5)
        self.pwcnet = PWCNet()
        self.feature_extractor = Vgg19Features()
        self.feature_extractor.load_pretrained_weights()

    def _get_forward(self, images_0, images_1, t, reuse_variables=tf.AUTO_REUSE):
        """
        :param images_0: Tensor of shape [batch_size, H, W, 3].
        :param images_1: Tensor of shape [batch_size, H, W, 3].
        :param t: Float. Specifies the interpolation point (i.e 0 for images_0, 1 for images_1).
        :return: interpolated: The interpolated image. Tensor of shape [batch_size, H, W, 3].
                 warped_0_1: Image and features from image_0 forward-flowed towards image_b, before synthesis.
                             The first 3 channels are the image.
                 warped_1_0: Image and features from image_b forward-flowed towards image_a, before synthesis.
                             The first 3 channels are the image.
                 flow_0_1: Flow from images 0 to 1 (centered at images 0).
                 flow_1_0: Flow from images 1 to 0 (centered at images 1).
        """
        self.enclosing_scope = tf.get_variable_scope()
        with tf.variable_scope(self.name, reuse=reuse_variables):
            batch_size = tf.shape(images_0)[0]
            from_frames = tf.concat([images_0, images_1], axis=0)
            to_frames = tf.concat([images_1, images_0], axis=0)
            all_contexts = self.feature_extractor.get_context_features(from_frames)

            # TODO: Add instance normalization. Described in 3.3 of https://arxiv.org/pdf/1803.10967.pdf.

            # Get images 0->1 and images 1->0 flows from PWCNet.
            # TODO: Migrate to pwcnet.get_bidirectional.
            all_flows, _ = self.pwcnet.get_forward(from_frames, to_frames, reuse_variables=reuse_variables)
            flow_0_1 = all_flows[:batch_size]
            flow_1_0 = all_flows[batch_size:]

            features_a = tf.concat([images_0, all_contexts[:batch_size]], axis=-1)
            features_b = tf.concat([images_1, all_contexts[batch_size:]], axis=-1)
            all_features = tf.concat([features_a, features_b], axis=0)
            all_warp_flows = tf.concat([t * flow_0_1, (1.0 - t) * flow_1_0], axis=0)

            # Warp images and their contexts from images 0->1 and from images 1->0.
            all_warped = forward_warp(all_features, all_warp_flows)
            warped_0_1 = tf.stop_gradient(all_warped[:batch_size])
            warped_1_0 = tf.stop_gradient(all_warped[batch_size:])

            # Feed into GridNet for final synthesis.
            warped_combined = tf.concat([warped_0_1, warped_1_0], axis=-1)
            synthesized, _, _, _ = self.gridnet.get_forward(warped_combined, training=True)
            return synthesized, warped_0_1, warped_1_0, flow_0_1, flow_1_0

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
            batch_size = tf.shape(prediction)[0]
            combined = tf.concat([prediction, expected], axis=0)
            all_features = self.feature_extractor.get_perceptual_features(combined)
            return tf.reduce_mean(tf.squared_difference(all_features[:batch_size], all_features[batch_size:]))

    def _get_l1_loss(self, prediction, expected):
        return tf.reduce_mean(tf.abs(prediction - expected))

    def _get_laplacian_loss(self, prediction, expected):
        with tf.name_scope('laplacian_loss'):
            batch_size = tf.shape(prediction)[0]
            combined = tf.concat([prediction, expected], axis=0)
            pyrs, _, _ = self.laplacian_pyramid.get_forward(combined)
            loss = 0
            for i in range(len(pyrs)):
                loss += 2 ** i * tf.reduce_sum(tf.abs(pyrs[i][:batch_size] - pyrs[i][batch_size:]))
            return loss
