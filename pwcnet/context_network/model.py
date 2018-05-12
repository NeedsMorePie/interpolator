import tensorflow as tf
from common.models import ConvNetwork


class ContextNetwork(ConvNetwork):
    def __init__(self, name='context_network', layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 regularizer=None):
        """
        Context network -- usually has 6 layer + a delta optical flow output layer.
        The delta optical flow is added to the inputted optical flow.
        :param name: Str. For variable scoping.
        :param layer_specs: See parent class.
        :param activation_fn: Tensorflow activation function.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        """
        super().__init__(layer_specs=layer_specs,
                         activation_fn=activation_fn, regularizer=regularizer, padding='SAME')

        self.name = name
        if layer_specs is None:
            # PWC-Net default.
            self.layer_specs = [[3, 128, 1, 1],
                                [3, 128, 2, 1],
                                [3, 128, 4, 1],
                                [3, 96, 8, 1],
                                [3, 64, 16, 1],
                                [3, 32, 1, 1]]
        else:
            self.layer_specs = layer_specs

    def get_forward(self, features, optical_flow, reuse_variables=False):
        """
        features  optical_flow
            \           /  \
              [LAYER 0]    |
                 ...       |
             [LAYER N-1]   |
                  |        |
            [Output layer] |
                  |        |
              delta_flow   |
                  +       /
             optical_flow
                  |
             final_output
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :param optical_flow: Tensor. Optical flow of shape [batch_size, H, W, 2].
        :param reuse_variables: Bool. Whether to reuse the variables.
        :return: final_flow: optical flow of shape [batch_size, H, W, 2].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            # Initial input has shape [batch_size, H, W, num_features + 2].
            initial_input = tf.concat([features, optical_flow], axis=-1)

            previous_output, layer_outputs = self._get_conv_tower(initial_input)

            # Create the last convolution layer that outputs the delta flow.
            previous_output = tf.layers.conv2d(inputs=previous_output,
                                               filters=2,
                                               kernel_size=[3, 3],
                                               padding='SAME',
                                               dilation_rate=(1, 1),
                                               activation=None,
                                               kernel_regularizer=self.regularizer,
                                               bias_regularizer=self.regularizer,
                                               name='delta_flow_conv')
            layer_outputs.append(previous_output)
            delta_flow = previous_output

            # Final output is the delta
            final_flow = delta_flow + optical_flow
            return final_flow, layer_outputs
