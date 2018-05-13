import tensorflow as tf
from common.models import ConvNetwork


def get_dropped_out_one(dropout_rate, training):
    one = tf.constant(1.0)
    return tf.layers.dropout(one, rate=dropout_rate, training=training)


class LateralConnection(ConvNetwork):
    def __init__(self, name, layer_specs,
                 activation_fn=tf.nn.leaky_relu,
                 use_batch_norm=False,
                 total_dropout_rate=0.0,
                 regularizer=None):
        """
        :param name: Str. For variable scoping.
        :param layer_specs: Array of shape [num_layers, 2]. Constrained version of parent class' layer_specs.
                            The second dimension consists of [num_output_features, dilation].
        :param activation_fn: Tensorflow activation function. This will not be applied on the last convolutional layer.
        :param use_batch_norm: Whether to use batch normalization.
        :param total_dropout_rate: A value of 1.0 will always zero-out the output of this block, and 0.0 will keep it.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        """
        full_layer_specs = []
        for i, layer_spec in enumerate(layer_specs):
            full_layer_spec = [3, layer_spec[0], layer_spec[1], 1]
            full_layer_specs.append(full_layer_spec)

        super().__init__(layer_specs=full_layer_specs,
                         activation_fn=activation_fn, last_activation_fn=None,
                         regularizer=regularizer, padding='SAME')

        self.name = name
        self.total_dropout_rate = total_dropout_rate

    def get_forward(self, features, reuse_variables=False, training=False):
        """
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :param reuse_variables: Bool. Whether to reuse the variables.
        :return: Tensor. Feature map of shape [batch_size, H, W, num_features].
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):

            # Pass through resolution preserving convolutions, with skip connection at the end.
            previous_output = self.activation_fn(features)
            previous_output, layer_outputs = self._get_conv_tower(previous_output)
            dropped_out = previous_output * get_dropped_out_one(self.total_dropout_rate, training=training)
            return dropped_out + features


class DownSamplingConnection(ConvNetwork):
    def __init__(self, name, layer_specs,
                 activation_fn=tf.nn.leaky_relu,
                 use_batch_norm=False,
                 regularizer=None):
        """
        :param name: Str. For variable scoping.
        :param layer_specs: Array of shape [num_layers, 2]. Constrained version of parent class' layer_specs.
                            The second dimension consists of [num_output_features, dilation].
        :param activation_fn: Tensorflow activation function. This will not be applied on the last convolutional layer.
        :param use_batch_norm: Whether to use batch normalization.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        """

        full_layer_specs = []
        for i, layer_spec in enumerate(layer_specs):
            stride = 2 if i == 0 else 1
            full_layer_spec = [3, layer_spec[0], layer_spec[1], stride]
            full_layer_specs.append(full_layer_spec)

        super().__init__(layer_specs=full_layer_specs,
                         activation_fn=activation_fn, last_activation_fn=None,
                         regularizer=regularizer, padding='SAME')

        self.name = name

    def get_forward(self, features, reuse_variables=False):
        """
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :param reuse_variables: Bool. Whether to reuse the variables.
        :return: Tensor. Feature map of shape [batch_size, H, W, num_features].
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            previous_output = self.activation_fn(features)
            final_output, layer_outputs = self._get_conv_tower(previous_output)
            return final_output


class UpSamplingConnection(ConvNetwork):
    def __init__(self, name, layer_specs,
                 activation_fn=tf.nn.leaky_relu,
                 use_batch_norm=False,
                 regularizer=None):
        """
        :param name: Str. For variable scoping.
        :param layer_specs: Array of shape [num_layers, 2]. Constrained version of parent class' layer_specs.
                            The second dimension consists of [num_output_features, dilation].
        :param activation_fn: Tensorflow activation function. This will not be applied on the last convolutional layer.
        :param use_batch_norm: Whether to use batch normalization.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        """
        full_layer_specs = []
        for i, layer_spec in enumerate(layer_specs):
            full_layer_spec = [3, layer_spec[0], layer_spec[1], 1]
            full_layer_specs.append(full_layer_spec)

        super().__init__(layer_specs=full_layer_specs,
                         activation_fn=activation_fn, last_activation_fn=None,
                         regularizer=regularizer, padding='SAME')

        self.name = name

    def get_forward(self, features, reuse_variables=False, training=False):
        """
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :param reuse_variables: Bool. Whether to reuse the variables.
        :return: Tensor. Feature map of shape [batch_size, H, W, num_features].
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):

            # Up-sample feature images.
            cur_height, cur_width = tf.shape(features)[1], tf.shape(features)[2]
            new_height, new_width = 2 * cur_height, 2 * cur_width
            previous_output = tf.image.resize_bilinear(features, (new_height, new_width))

            # Pass through resolution preserving convolutions.
            previous_output = self.activation_fn(previous_output)
            final_output, layer_outputs = self._get_conv_tower(previous_output)
            return final_output

