import tensorflow as tf
from common.models import ConvNetwork


class ContextNetwork(ConvNetwork):
    def __init__(self, name='context_network', layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 regularizer=None, dense_net=False):
        """
        Context network -- usually has 6 layer + a delta optical flow output layer.
        The delta optical flow is added to the inputted optical flow.
        :param name: Str. For variable scoping.
        :param layer_specs: See parent class.
        :param activation_fn: Tensorflow activation function.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        :param dense_net: Bool.
        """
        super().__init__(name=name, layer_specs=layer_specs,
                         activation_fn=activation_fn, last_activation_fn=None,
                         regularizer=regularizer, padding='SAME', dense_net=dense_net)

        if layer_specs is None:
            # PWC-Net default.
            self.layer_specs = [[3, 128, 1, 1],
                                [3, 128, 2, 1],
                                [3, 128, 4, 1],
                                [3, 96, 8, 1],
                                [3, 64, 16, 1],
                                [3, 32, 1, 1],
                                [3, 2, 1, 1]]  # last_activation_fn is linear.
        else:
            self.layer_specs = layer_specs

    def get_forward(self, features, optical_flow, reuse_variables=tf.AUTO_REUSE):
        """
        features  optical_flow
            \           /  \
              [LAYER 0]    |
                 ...       |
             [LAYER N-1]   |
                  |        |
              delta_flow   |
                  +       /
             optical_flow
                  |
             final_output
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :param optical_flow: Tensor. Optical flow of shape [batch_size, H, W, 2].
        :param reuse_variables: tf reuse option. i.e. tf.AUTO_REUSE.
        :return: final_flow: optical flow of shape [batch_size, H, W, 2].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            # Initial input has shape [batch_size, H, W, num_features + 2].
            initial_input = tf.concat([features, optical_flow], axis=-1)

            delta_flow, layer_outputs = self._get_conv_tower(initial_input)

            # Final output is the delta
            final_flow = delta_flow + optical_flow
            return final_flow, layer_outputs
