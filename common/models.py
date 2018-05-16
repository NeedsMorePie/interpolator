import tensorflow as tf
from utils.misc import print_tensor_shape

_default = object()

class ConvNetwork:
    def __init__(self, layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 last_activation_fn=_default,
                 regularizer=None, padding='SAME', dense_net=False):
        """
        Generic conv-net
        :param name: Str. For variable scoping.
        :param layer_specs: Array of shape [num_layers, 4].
                            The second dimension consists of [kernel_size, num_output_features, dilation, stride].
        :param activation_fn: Tensorflow activation function.
        :param last_activation_fn: Tensorflow activation function. Applied after the final convolution of activation_fn,
                                   in place of activation_fn. Defaults to the value of activation_fn.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        :param padding: Str. Either 'SAME' or 'VALID' case insensitive.
        :param dense_net: Bool. If true, then it is expected that all layers have the same width and height.
        """
        self.layer_specs = layer_specs
        self.activation_fn = activation_fn
        self.regularizer = regularizer
        self.padding = padding
        self.dense_net = dense_net

        if last_activation_fn == _default:
            self.last_activation_fn = self.activation_fn
        else:
            self.last_activation_fn = last_activation_fn

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

            is_last_layer = i == len(self.layer_specs) - 1
            activation_fn = self.last_activation_fn if is_last_layer else self.activation_fn
            if self.dense_net and i != 0:
                # Dense-net layer input consists of all previous layer outputs.
                assert previous_output == layer_outputs[-1]
                inputs = tf.concat(layer_outputs, axis=-1)
            else:
                inputs = previous_output

            # Create the convolution layer.
            previous_output = tf.layers.conv2d(inputs=inputs,
                                               filters=num_output_features,
                                               kernel_size=[kernel_size, kernel_size],
                                               strides=(stride, stride),
                                               padding='SAME',
                                               dilation_rate=(dilation, dilation),
                                               activation=None,
                                               kernel_regularizer=self.regularizer,
                                               bias_regularizer=self.regularizer,
                                               name='conv_' + str(i))

            if activation_fn is not None:
                previous_output = activation_fn(previous_output)

            layer_outputs.append(previous_output)

        final_output = previous_output
        return final_output, layer_outputs

