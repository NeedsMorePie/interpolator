import tensorflow as tf
from common.models import RestorableNetwork
from pwcnet.estimator_network.model import EstimatorNetwork
from pwcnet.context_network.model import ContextNetwork
from pwcnet.feature_pyramid_network.model import FeaturePyramidNetwork
from pwcnet.losses.loss import create_multi_level_loss, l2_diff, lq_diff, create_multi_level_unflow_loss
from tensorflow.contrib.layers import l2_regularizer


VERBOSE = False


class PWCNet(RestorableNetwork):
    def __init__(self, name='pwc_net', regularizer=l2_regularizer(4e-4),
                 flow_layer_loss_weights=None, flow_scaling=0.05, search_range=4):
        """
        :param name: Str.
        :param regularizer: Tf regularizer.
        :param flow_layer_loss_weights: List of floats. Corresponds to the weight of a loss for a flow at some layer.
                                        i.e. flow_layer_loss_weights[0] corresponds to previous_flows[0].
        :param flow_scaling: In the PWC-Net paper, ground truth is scaled by this amount to normalize the flows.
        :param search_range: The search range to use for the cost volume layer.
        """
        super().__init__(name=name)

        self.regularizer = regularizer
        self.flow_scaling = flow_scaling

        if flow_layer_loss_weights is None:
            self.flow_layer_loss_weights = [0.32, 0.08, 0.02, 0.01, 0.002, 0.003]
        else:
            self.flow_layer_loss_weights = flow_layer_loss_weights

        # Number of times the flow is estimated and refined.
        # If this number changes, then the feature_pyramid needs to be reconfigured.
        self.num_feature_levels = 6
        # This is the output feature level. This can be anywhere between [1, self.num_feature_levels].
        self.output_level = 2
        # A range that counts down from [self.num_flow_estimates, self.output_level] inclusive.
        self.iter_range = range(self.num_feature_levels, self.output_level - 1, -1)

        self.feature_pyramid = FeaturePyramidNetwork(regularizer=self.regularizer)
        self.estimator_networks = [EstimatorNetwork(name='estimator_network_' + str(i),
                                                    regularizer=self.regularizer,
                                                    search_range=search_range)
                                   for i in self.iter_range]
        self.context_network = ContextNetwork(regularizer=self.regularizer)

    def get_forward(self, image_a, image_b, reuse_variables=tf.AUTO_REUSE):
        """
        :param image_a: Tensor of shape [batch_size, H, W, 3].
        :param image_b: Tensor of shape [batch_size, H, W, 3].
        :return: final_flow: up-sampled final flow.
                 previous_flows: all previous flow outputs of the estimator networks and the context network.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            batch_size = tf.shape(image_a)[0]
            img_height = tf.shape(image_a)[1]
            img_width = tf.shape(image_a)[2]
            # Siamese networks (i.e. image_a and image_b are fed through the same network with shared weights).
            # Implemented by combining the the image_a and image_b batches.
            images_a_b = tf.concat([image_a, image_b], axis=0)
            _, features = self.feature_pyramid.get_forward(images_a_b, reuse_variables=reuse_variables)

            # The initial flow is None. The estimator not do warping if flow is None.
            # It is refined at each feature level.
            previous_flow = None
            previous_flows = []
            # Intermediate features from the previous estimator network.
            previous_estimator_features = None

            # Counts down from [self.num_flow_estimates, self.output_level] inclusive.
            for i in self.iter_range:
                if VERBOSE:
                    print('Creating estimator at level', i)
                # Get the features at this level.
                features_a_n, features_b_n = self._get_image_features_for_level(features, i, batch_size)

                # Setup the previous flow and feature map for input into the estimator network at this level.
                H, W = tf.shape(features_a_n)[1], tf.shape(features_a_n)[2]
                resized_flow, pre_warp_scaling = self._create_resized_flow_for_next_estimator(
                    previous_flow, H, W, img_height, name='resize_previous_flow' + str(i))
                upsampled_previous_features = self._create_upsampled_features_for_next_estimator(
                    previous_estimator_features, name='deconv_estimator_features_' + str(i))

                # Get the estimator network.
                estimator_network = self.estimator_networks[self.num_feature_levels - i]
                if VERBOSE:
                    print('Getting forward ops for', estimator_network.name)
                previous_flow, estimator_outputs, dense_outputs = estimator_network.get_forward(
                    features_a_n, features_b_n, resized_flow, upsampled_previous_features,
                    pre_warp_scaling=pre_warp_scaling, reuse_variables=reuse_variables)
                previous_flows.append(previous_flow)
                assert estimator_outputs[-1] == previous_flow
                # Get the previous_estimator_features differently depending on whether the estimator is dense.
                if estimator_network.dense_net:
                    assert len(dense_outputs) > 1
                    previous_estimator_features = dense_outputs[-2]
                else:
                    assert len(estimator_outputs) > 1
                    previous_estimator_features = estimator_outputs[-2]

                # Last level gets the context-network treatment.
                if i == self.output_level:
                    if VERBOSE:
                        print('Getting forward ops for context network.')
                    # Features are the second to last output of the estimator network.
                    previous_flow, _ = self.context_network.get_forward(
                        previous_estimator_features, previous_flow, reuse_variables=reuse_variables)
                    previous_flows.append(previous_flow)

            final_flow = tf.image.resize_bilinear(previous_flow, [img_height, img_width])
            final_flow = tf.divide(final_flow, self.flow_scaling, name='final_flow')
            return final_flow, previous_flows

    def get_bidirectional(self, image_a, image_b, reuse_variables=tf.AUTO_REUSE):
        """
        Gets the bidirectional flow using a siamese PWC Net.
        :param image_a: Tensor of shape [batch_size, H, W, 3].
        :param image_b: Tensor of shape [batch_size, H, W, 3].
        :return: final_forward_flow: up-sampled final forward flow.
                 final_backward_flow: up-sampled final backward flow.
                 previous_forward_flows: all previous flow outputs of the estimator networks and the context network.
                 previous_backward_flows: all previous flow outputs of the estimator networks and the context network.
        """
        batch_size = tf.shape(image_a)[0]
        input_stack_a = tf.concat([image_a, image_b], axis=0)
        input_stack_b = tf.concat([image_b, image_a], axis=0)
        final_flow, previous_flows = self.get_forward(input_stack_a, input_stack_b, reuse_variables=reuse_variables)
        final_forward_flow = final_flow[0:batch_size, ...]
        final_backward_flow = final_flow[batch_size:, ...]
        previous_forward_flows = []
        previous_backward_flows = []
        for previous_flow in previous_flows:
            previous_forward_flows.append(previous_flow[0:batch_size, ...])
            previous_backward_flows.append(previous_flow[batch_size:, ...])

        return final_forward_flow, final_backward_flow, previous_forward_flows, previous_backward_flows

    def _get_image_features_for_level(self, feature_levels, level, batch_size):
        """
        Extracts the features for image_a and image_b from the feature pyramid.
        :param feature_levels: List of features from the feature pyramid.
        :param level: Int.
        :param batch_size: Scalar tensor.
        :return: features_a_n, features_b_n: Tensors of shape [batch_size, height, width, channels].
        """
        features_n = feature_levels[self.feature_pyramid.get_c_n_idx(level)]
        with tf.name_scope('features_a_' + str(level)):
            features_a_n = features_n[0:batch_size, ...]
        with tf.name_scope('features_b_' + str(level)):
            features_b_n = features_n[batch_size:, ...]
        return features_a_n, features_b_n

    def _create_resized_flow_for_next_estimator(self, previous_flow, desired_height, desired_width, img_height, name):
        """
        Scales the flow for the next estimator network. Also computes the pre-warp scaling to denormalize the flow.
        :param previous_flow: Tensor of shape [batch, height, width, 2] or None.
        :param desired_height: Int or tensor. Height to scale to.
        :param desired_width: Int or tensor. Width to scale to.
        :param img_height: Int or tensor. Height of the network's input image.
        :param name: Str.
        :return: resized_flow: Tensor of shape [batch, desired_height, desired_width, 2]. None if previous_flow is None.
                 pre_warp_scaling: Scalar tensor. 1.0 if previous_flow is None.
        """
        pre_warp_scaling = 1.0
        resized_flow = None
        if previous_flow is not None:
            # The original scale flows at all layers is the same as the scale of the ground truth.
            dimension_scaling = tf.cast(desired_height, tf.float32) / tf.cast(img_height, tf.float32)
            pre_warp_scaling = dimension_scaling / self.flow_scaling
            # Upsample to the size of the current layer.
            resized_flow = tf.image.resize_bilinear(previous_flow, [desired_height, desired_width], name=name)
        return resized_flow, pre_warp_scaling

    def _create_upsampled_features_for_next_estimator(self, features, name):
        """
        Upsamples a feature map and encodes it into 2 channels.
        :param features: Tensor of shape [batch, height, width, channels] or None.
        :param name: Str.
        :return: Tensor of shape [batch, height * 2, width * 2, 2] or None if features is none.
        """
        upsampled_features = None
        if features is not None:
            upsampled_features = tf.layers.conv2d_transpose(
                features, filters=2, kernel_size=4, strides=2, padding='same', use_bias=True,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, name=name)
        return upsampled_features

    def _get_loss(self, previous_flows, expected_flow, diff_fn):
        """
        :param previous_flows: List of previous outputs from the PWC-Net forward pass.
        :param expected_flow: Tensor of shape [batch_size, H, W, 2]. These are the ground-truth labels.
        :param diff_fn: Function to diff 2 tensors.
        :return: Tf scalar loss term, and an array of all the inidividual loss terms.
        """

        total_loss, layer_losses = create_multi_level_loss(
            expected_flow=expected_flow, flows=previous_flows, flow_scaling=self.flow_scaling,
            flow_layer_loss_weights=self.flow_layer_loss_weights, diff_fn=diff_fn)

        # Add the regularization loss.
        total_loss += tf.add_n(tf.losses.get_regularization_losses(scope=self.name))

        return total_loss, layer_losses

    def get_training_loss(self, previous_flows, expected_flow):
        """
        Uses an L2 diffing loss.
        :return: Tf scalar loss term, and an array of all the inidividual loss terms.
        """
        with tf.name_scope('training_loss'):
            return self._get_loss(previous_flows, expected_flow, l2_diff)

    def get_fine_tuning_loss(self, previous_flows, expected_flow, q=0.4, epsilon=0.01):
        """
        Uses an Lq diffing loss.
        :return: Tf scalar loss term, and an array of all the inidividual loss terms.
        """
        with tf.name_scope('fine_tuning_loss'):
            def lq_diff_with_args(a, b):
                return lq_diff(a, b, q=q, epsilon=epsilon)
            return self._get_loss(previous_flows, expected_flow, lq_diff_with_args)

    def get_unflow_training_loss(self, image_a, image_b, previous_forward_flows, previous_backward_flows):
        """
        :param image_a: Tensor of shape [B, H, W, 3].
        :param image_b: Tensor of shape [B, H, W, 3].
        :param previous_forward_flows: List of flow tensors at different resolution levels.
        :param previous_backward_flows: List of flow tensors at different resolution levels.
        :return: total_loss: Scalar tensor. Sum off all weighted level losses.
                 layer_losses: List of scalar tensors. Weighted loss at each level.
                 forward_occlusion_masks: List of tensors of shape [B, H, W, 1].
                 backward_occlusion_masks: List of tensors of shape [B, H, W, 1].
                 layer_losses_detailed: List of dicts. Each dict contains the individual fully weighted losses.
        """
        return create_multi_level_unflow_loss(image_a, image_b, previous_forward_flows, previous_backward_flows,
                                              self.flow_scaling)
