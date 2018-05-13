import tensorflow as tf
from pwcnet.estimator_network.model import EstimatorNetwork
from pwcnet.context_network.model import ContextNetwork
from pwcnet.feature_pyramid_network.model import FeaturePyramidNetwork
from tensorflow.contrib.layers import l2_regularizer


VERBOSE = False


class PWCNet:
    def __init__(self, name='pwc_net', regularizer=l2_regularizer(1e-4)):
        self.name = name
        self.regularizer = regularizer

        # Number of times the flow is estimated and refined.
        # If this number changes, then the feature_pyramid needs to be reconfigured.
        self.num_feature_levels = 6
        # This is the output feature level. This can be anywhere between [1, self.num_feature_levels].
        self.output_level = 2
        # A range that counts down from [self.num_flow_estimates, self.output_level] inclusive.
        self.iter_range = range(self.num_feature_levels, self.output_level - 1, -1)

        self.feature_pyramid = FeaturePyramidNetwork(regularizer=self.regularizer)
        self.estimator_networks = [EstimatorNetwork(name='estimator_network_' + str(i), regularizer=self.regularizer)
                                   for i in self.iter_range]
        self.context_network = ContextNetwork(regularizer=self.regularizer)

    def get_forward(self, image_a, image_b):
        """
        :param image_a: Tensor of shape [batch_size, H, W, 3].
        :param image_b: Tensor of shape [batch_size, H, W, 3].
        :return: final_flow: up-sampled final flow.
                 previous_flows: all previous flow outputs of the estimator networks and the context network.
        """
        with tf.variable_scope(self.name):
            img_height = tf.shape(image_a)[1]
            img_width = tf.shape(image_a)[2]
            # Siamese networks (i.e. image_a and image_b are fed through the same network with shared weights).
            _, features_a = self.feature_pyramid.get_forward(image_a, reuse_variables=tf.AUTO_REUSE)
            _, features_b = self.feature_pyramid.get_forward(image_b, reuse_variables=tf.AUTO_REUSE)

            # The initial flow will be initialized with zeros.
            # It is refined at each feature level.
            previous_flow = None
            previous_flows = []

            # Counts down from [self.num_flow_estimates, self.output_level] inclusive.
            for i in self.iter_range:
                if VERBOSE:
                    print('Creating estimator at level', i)
                # Get the features at this level.
                features_a_n = features_a[self.feature_pyramid.get_c_n(i)]
                features_b_n = features_b[self.feature_pyramid.get_c_n(i)]

                B = tf.shape(features_a_n)[0]
                H = tf.shape(features_a_n)[1]
                W = tf.shape(features_a_n)[2]
                if previous_flow is None:
                    previous_flow = tf.zeros(shape=[B, H, W, 2], dtype=tf.float32)
                else:
                    # Upsample to the size of the current layer.
                    previous_flow = tf.image.resize_images(previous_flow, [H, W],
                                                           method=tf.image.ResizeMethod.BILINEAR)

                # Get the estimator network.
                estimator_network = self.estimator_networks[self.num_feature_levels - i]
                if VERBOSE:
                    print('Getting forward ops for', estimator_network.name)
                previous_flow, estimator_outputs = estimator_network.get_forward(
                    features_a_n, features_b_n, previous_flow)
                previous_flows.append(previous_flow)

                if i == self.output_level:
                    if VERBOSE:
                        print('Getting forward ops for context network.')
                    assert estimator_outputs[-1] == previous_flow
                    # Features are the second to last output of the estimator network.
                    previous_flow, context_outputs = self.context_network.get_forward(
                        estimator_outputs[-2], previous_flow)
                    previous_flows.append(previous_flow)

            final_flow = tf.image.resize_images(previous_flow, [img_height, img_width],
                                                method=tf.image.ResizeMethod.BILINEAR)
            return final_flow, previous_flows
