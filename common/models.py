import tensorflow as tf


class ConvNetwork:
    def __init__(self, layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 regularizer=None, padding='SAME'):
        """
        Generic conv-net
        :param name: Str. For variable scoping.
        :param layer_specs: Array of shape [num_layers, 3].
                            The second dimension consists of [kernel_size, num_output_features, dilation, stride].
        :param activation_fn: Tensorflow activation function.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        :param padding: Str. Either 'SAME' or 'VALID' case insensitive.
        """
        self.layer_specs = layer_specs
        self.activation_fn = activation_fn
        self.regularizer = regularizer
        self.padding = padding

    def _get_conv_tower(self, features):
        """
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :return: final_output: tensor of shape [batch_size, H, W, num_output_features].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        layer_outputs = []

        # Create the network layers.
        previous_output = features
        for i, layer_spec in enumerate(self.layer_specs):
            # Get specs.
            kernel_size = layer_spec[0]
            num_output_features = layer_spec[1]
            dilation = layer_spec[2]
            stride = layer_spec[3]

            # Create the convolution layer.
            previous_output = tf.layers.conv2d(inputs=previous_output,
                                               filters=num_output_features,
                                               kernel_size=[kernel_size, kernel_size],
                                               strides=(stride, stride),
                                               padding='SAME',
                                               dilation_rate=(dilation, dilation),
                                               activation=self.activation_fn,
                                               kernel_regularizer=self.regularizer,
                                               bias_regularizer=self.regularizer,
                                               name='conv_' + str(i))
            layer_outputs.append(previous_output)

        final_output = previous_output
        return final_output, layer_outputs

