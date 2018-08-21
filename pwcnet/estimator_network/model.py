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
        super().__init__(name=name, layer_specs=layer_specs,
                         activation_fn=activation_fn, last_activation_fn=None,
                         regularizer=regularizer, padding='SAME', dense_net=dense_net)

        if layer_specs is None:
            # PWC-Net default.
            self.layer_specs = [[3, 128, 1, 1],
                                [3, 128, 1, 1],
                                [3, 96, 1, 1],
                                [3, 64, 1, 1],
                                [3, 32, 1, 1],
                                [3, 2, 1, 1]]  # last_activation_fn is linear.
        else:
            self.layer_specs = layer_specs

        self.search_range = search_range

    def get_forward(self, features1, features2, optical_flow, pre_warp_scaling=1.0, reuse_variables=tf.AUTO_REUSE):
        """
        features1   features2  optical_flow
              \         \           /  \
               \        [WARP_LAYER]    |
                \             |         |
                 -------[COST_VOLUME]   |
                  \           |        /
                   -------[LAYER 0]---
                             ...
                         [LAYER N-1]
                              |
                         final_output
        :param features1: Tensor. Feature map of shape [batch_size, H, W, num_features]. Time = 0.
        :param features2: Tensor. Feature map of shape [batch_size, H, W, num_features]. Time = 1.
        :param optical_flow: Tensor. Optical flow of shape [batch_size, H, W, 2].
        :param pre_warp_scaling: Tensor or scalar. Scaling to be applied right before warping.
        :param reuse_variables: tf reuse option. i.e. tf.AUTO_REUSE.
        :return: final_flow: optical flow of shape [batch_size, H, W, 2].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            # Warp layer.
            warped = warp_via_flow(features2, optical_flow * pre_warp_scaling)

            # Cost volume layer.
            cv = cost_volume(features1, warped, search_range=self.search_range)

            # CNN layers.
            # Initial input has shape [batch_size, H, W, in_features + cv_size + 2]
            initial_input = tf.concat([features1, cv, optical_flow], axis=-1)
            final_output, layer_outputs = self._get_conv_tower(initial_input)

            # Prepend the warp and cv outputs.
            layer_outputs = [warped, cv] + layer_outputs

            return final_output, layer_outputs
