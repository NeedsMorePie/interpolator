import tensorflow as tf
from common.models import RestorableNetwork
from pwcnet.estimator_network.model import EstimatorNetwork
from pwcnet.context_network.model import ContextNetwork
from pwcnet.feature_pyramid_network.model import FeaturePyramidNetwork
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
            self.flow_layer_loss_weights = [0.32, 0.08, 0.02, 0.01, 0.001, 0.005]
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
            # Features from the second last layer of the previous estimator network.
            previous_estimator_feature = None

            # Counts down from [self.num_flow_estimates, self.output_level] inclusive.
            for i in self.iter_range:
                if VERBOSE:
                    print('Creating estimator at level', i)
                # Get the features at this level.
                features_n = features[self.feature_pyramid.get_c_n_idx(i)]
                with tf.name_scope('features_a_' + str(i)):
                    features_a_n = features_n[0:batch_size, ...]
                with tf.name_scope('features_b_' + str(i)):
                    features_b_n = features_n[batch_size:, ...]

                # Setup the previous flow and feature map for input into the estimator network at this level.
                with tf.name_scope('height_' + str(i)):
                    H = tf.shape(features_a_n)[1]
                with tf.name_scope('width_' + str(i)):
                    W = tf.shape(features_a_n)[2]
                pre_warp_scaling = 1.0
                if previous_flow is not None:
                    # The original scale flows at all layers is the same as the scale of the ground truth.
                    dimension_scaling = tf.cast(H, tf.float32) / tf.cast(img_height, tf.float32)
                    pre_warp_scaling = dimension_scaling / self.flow_scaling
                    # Upsample to the size of the current layer.
                    with tf.name_scope('previous_flow_' + str(i)):
                        previous_flow = tf.image.resize_images(previous_flow, [H, W],
                                                               method=tf.image.ResizeMethod.BILINEAR)
                    with tf.name_scope('previous_estimator_feature_' + str(i)):
                        previous_estimator_feature = tf.image.resize_images(previous_estimator_feature, [H, W],
                                                                            method=tf.image.ResizeMethod.BILINEAR)

                # Get the estimator network.
                estimator_network = self.estimator_networks[self.num_feature_levels - i]
                if VERBOSE:
                    print('Getting forward ops for', estimator_network.name)
                previous_flow, estimator_outputs = estimator_network.get_forward(
                    features_a_n, features_b_n, previous_flow, previous_estimator_feature,
                    pre_warp_scaling=pre_warp_scaling, reuse_variables=reuse_variables)
                previous_flows.append(previous_flow)
                assert estimator_outputs[-1] == previous_flow
                previous_estimator_feature = estimator_outputs[-2]

                # Last level gets the context-network treatment.
                if i == self.output_level:
                    if VERBOSE:
                        print('Getting forward ops for context network.')
                    # Features are the second to last output of the estimator network.
                    previous_flow, context_outputs = self.context_network.get_forward(
                        tf.concat(estimator_outputs, axis=-1), previous_flow, reuse_variables=reuse_variables)
                    previous_flows.append(previous_flow)

            final_flow = tf.image.resize_images(previous_flow, [img_height, img_width],
                                                method=tf.image.ResizeMethod.BILINEAR)
            final_flow /= self.flow_scaling
            return final_flow, previous_flows

    def _get_loss(self, previous_flows, expected_flow, diff_fn):
        """
        :param previous_flows: List of previous outputs from the PWC-Net forward pass.
        :param expected_flow: Tensor of shape [batch_size, H, W, 2]. These are the ground-truth labels.
        :param diff_fn: Function to diff 2 tensors.
        :return: Tf scalar loss term, and an array of all the inidividual loss terms.
        """
        total_loss = tf.constant(0.0, dtype=tf.float32)
        scaled_gt = expected_flow * self.flow_scaling

        layer_losses = []
        for i, previous_flow in enumerate(previous_flows):
            with tf.name_scope('layer_' + str(i) + '_loss'):
                H, W = tf.shape(previous_flow)[1], tf.shape(previous_flow)[2]

                # Ground truth needs to be resized to match the size of the previous flow.
                resized_scaled_gt = tf.image.resize_images(scaled_gt, [H, W], method=tf.image.ResizeMethod.BILINEAR)

                # squared_difference has the shape [batch_size, H, W, 2].
                squared_difference = diff_fn(resized_scaled_gt, previous_flow)
                # Reduce sum in the last 3 dimensions, average over the batch, and apply the weight.
                weight = self.flow_layer_loss_weights[i]
                layer_loss = weight * tf.reduce_mean(tf.reduce_sum(squared_difference, axis=[1, 2, 3]))

                # Accumulate the total loss.
                layer_losses.append(layer_loss)
            total_loss += layer_loss

        # Add the regularization loss.
        total_loss += tf.add_n(tf.losses.get_regularization_losses(scope=self.name))
        return total_loss, layer_losses

    def get_training_loss(self, previous_flows, expected_flow):
        """
        Uses an L2 diffing loss.
        :return: Tf scalar loss term, and an array of all the inidividual loss terms.
        """
        with tf.name_scope('training_loss'):
            def l2_diff(a, b):
                return tf.square(a-b)
            return self._get_loss(previous_flows, expected_flow, l2_diff)

    def get_fine_tuning_loss(self, previous_flows, expected_flow, q=0.4, epsilon=0.01):
        """
        Uses an Lq diffing loss.
        :return: Tf scalar loss term, and an array of all the inidividual loss terms.
        """
        with tf.name_scope('fine_tuning_loss'):
            def lq_diff(a, b):
                return tf.pow(tf.abs(a-b) + epsilon, q)
            return self._get_loss(previous_flows, expected_flow, lq_diff)
