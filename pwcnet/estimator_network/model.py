import tensorflow as tf
from common.models import ConvNetwork
from pwcnet.cost_volume.cost_volume import cost_volume
from pwcnet.warp.warp import warp_via_flow


class EstimatorNetwork(ConvNetwork):
    def __init__(self, name='estimator_network', layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 regularizer=None, search_range=4, dense_net=True):
        """
        :param name: Str. For variable scoping.
        :param layer_specs: See parent class.
        :param activation_fn: Tensorflow activation function.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        :param dense_net: Bool. Default for PWC-Net is true.
        """
        super().__init__(layer_specs=layer_specs,
                         activation_fn=activation_fn, regularizer=regularizer, padding='SAME', dense_net=dense_net)

        self.name = name
        if layer_specs is None:
            # PWC-Net default.
            self.layer_specs = [[3, 128, 1, 1],
                                [3, 128, 1, 1],
                                [3, 96, 1, 1],
                                [3, 64, 1, 1],
                                [3, 32, 1, 1]]
        else:
            self.layer_specs = layer_specs

        self.search_range = search_range

    def get_forward(self, features1, features2, optical_flow, prewarp_scale=1.0, reuse_variables=tf.AUTO_REUSE):
        """
        features1   features2  optical_flow
              \         \           /
               \        [WARP_LAYER]
                \             |
                 -------[COST_VOLUME]
                  \           |
                   -------[LAYER 0]
                             ...
                         [LAYER N-1]
                              |
                        [Output layer]
                              |
                         final_output
        :param features1: Tensor. Feature map of shape [batch_size, H, W, num_features]. Time = 0.
        :param features2: Tensor. Feature map of shape [batch_size, H, W, num_features]. Time = 1.
        :param optical_flow: Tensor. Optical flow of shape [batch_size, H, W, 2].
        :param prewarp_scale: Float. Optical flow is scaled by this just before warping.
        :param reuse_variables: tf reuse option. i.e. tf.AUTO_REUSE.
        :return: final_flow: optical flow of shape [batch_size, H, W, 2].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            # Warp layer.
            warped = warp_via_flow(features2, optical_flow * prewarp_scale)

            # Cost volume layer.
            cv = cost_volume(features1, warped, search_range=self.search_range)

            # CNN layers.
            # Initial input has shape [batch_size, H, W, in_features + cv_size]
            initial_input = tf.concat([features1, cv], axis=-1)
            previous_output, layer_outputs = self._get_conv_tower(initial_input)

            if self.dense_net:
                # layer_outputs contains previous_output.
                assert previous_output == layer_outputs[-1]
                previous_output = tf.concat(layer_outputs, axis=-1)

            # Create the last convolution layer that outputs the delta flow.
            previous_output = tf.layers.conv2d(inputs=previous_output,
                                               filters=2,
                                               kernel_size=[3, 3],
                                               padding='SAME',
                                               dilation_rate=(1, 1),
                                               activation=None,
                                               kernel_regularizer=self.regularizer,
                                               bias_regularizer=self.regularizer,
                                               name='optical_flow_estimate')
            layer_outputs.append(previous_output)

            # Prepend the warp and cv outputs.
            layer_outputs = [warped, cv] + layer_outputs

            final_output = previous_output
            return final_output, layer_outputs
