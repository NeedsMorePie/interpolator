import tensorflow as tf
from utils.tf import print_tensor_shape, pelu, prelu
from context_interp.gridnet.connections.connections import UpSamplingConnection, DownSamplingConnection, LateralConnection


# Packaged 'activation' functions.
def batch_norm_with_prelu(x):
    x = tf.layers.batch_normalization(x)
    return prelu(x)


def batch_norm_with_relu(x):
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)


_default = object()


class GridNet:
    def __init__(self, channel_sizes, width,
                 name='gridnet',
                 num_output_channels=_default,
                 num_lateral_convs=2,
                 num_upsample_convs=2,
                 num_downsampling_convs=2,
                 use_batch_norm=False,
                 connection_dropout_rate=0.0,
                 regularizer=None):
        """
        See https://arxiv.org/pdf/1707.07958.pdf, and modifications made in https://arxiv.org/pdf/1803.10967.pdf.
        :param channel_sizes: List of channel sizes for rows. Height of the GridNet = len(channel_sizes).
        :param width: Width of the GridNet. Must be an even number to ensure symmetry.
        :param name: Str. For variable scoping.
        :param num_output_channels: Number. Defaults to channel_sizes[0].
        :param num_lateral_convs: Number of convolutions in each lateral connection.
        :param num_upsample_convs: Number of convolutions in each up-sampling connection.
        :param num_downsampling_convs: Number of convolutions in each down-sampling connection.
        :param use_batch_norm: Whether to use batch normalization.
        :param connection_dropout_rate: E.g if 0.5, drops out each connection (not individual neurons) with 50% chance.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.

        Example for height = 3, width = 4:

                       grid[0][0]                  grid[0][3]

        features +-------> +-------> +-------> +-------> +-------> output
                           |         |         ^         ^
                           |         |         |         |
                           |         |         |         |
                           |         |         |         |
                           v         v         |         |
                           +-------> +-------> +-------> +
                           |         |         ^         ^
                           |         |         |         |
                           |         |         |         |
                           |         |         |         |
                           v         v         |         |
                           +-------> +-------> +-------> +
        """
        
        height = len(channel_sizes)
        if height <= 0:
            raise ValueError('Height = len(channel_sizes) must be >= 1.')

        if width % 2 != 0:
            raise ValueError('Width must be an even number, to enforce GridNet symmetry.')

        if width <= 0:
            raise ValueError('Width must be non-zero.')

        self.width = width
        self.height = height
        self.channel_sizes = channel_sizes
        self.name = name
        self.num_lateral_convs = num_lateral_convs
        self.num_downsampling_convs = num_downsampling_convs
        self.num_upsample_convs = num_upsample_convs
        self.use_batch_norm = use_batch_norm
        self.connection_dropout_rate = connection_dropout_rate
        self.regularizer = regularizer

        if num_output_channels == _default:
            num_output_channels = self.channel_sizes[0]
        self.num_output_channels = num_output_channels

        # More settings
        if self.use_batch_norm:
            self.activation_fn = batch_norm_with_prelu
        else:
            self.activation_fn = prelu

        # Construct specs for connections.
        # Entry specs[i][j] is the spec for the jth convolution for any connection in the ith row.
        # An exception is made for the output lateral connection.
        self.output_spec = [[self.num_output_channels, 1] for j in range(self.num_lateral_convs)]
        self.lateral_specs = []
        self.upsample_specs = []
        self.downsample_specs = []
        for i in range(self.height):
            row_lateral_specs, row_upsample_specs, row_downsample_specs = [], [], []
            num_filters = channel_sizes[i]
            common_spec = [num_filters, 1]
            row_lateral_specs = [common_spec for j in range(self.num_lateral_convs)]

            if i > 0:
                row_downsample_specs = [common_spec for j in range(self.num_downsampling_convs)]

            if i < self.height - 1:
                row_upsample_specs = [common_spec for j in range(self.num_upsample_convs)]

            self.lateral_specs.append(row_lateral_specs)
            self.upsample_specs.append(row_upsample_specs)
            self.downsample_specs.append(row_downsample_specs)

    def get_forward(self, features, training=False, reuse_variables=False):
        """
        :param features: A Tensor. Input feature maps of shape [batch_size, H, W, num_features]
        :param training: A Bool. Whether the graph is to be constructed for training (dropout will be applied).
        :return: final_output: A Tensor. Will take on the same shape as param features.
                 node_outputs: A 2D list of Tensors. Represents the output at each grid node.
                 lateral_inputs: A 2D list of Tensors. Represents the lateral stream input at each grid node.
                 vertical_inputs: A 2D list of Tensors. Represents the vertical stream input at each grid node.
        """

        with tf.variable_scope(self.name, reuse=reuse_variables):
            node_outputs = [[None for x in range(self.width)] for y in range(self.height)]
            lateral_inputs = [[tf.constant(0.0) for x in range(self.width)] for y in range(self.height)]
            vertical_inputs = [[tf.constant(0.0) for x in range(self.width)] for y in range(self.height)]

            # First lateral connection.
            node_outputs[0][0] = self._process_rightwards(features, 0, 0, training=training)
            lateral_inputs[0][0] = node_outputs[0][0]

            # Connect first half (Down-sampling streams) by iterating to the right, and downwards.
            for i in range(self.height):
                for j in range(int(self.width / 2)):
                    if i == 0 and j == 0:
                        continue

                    top_output, left_output = 0, 0
                    if i > 0:
                        top_output = self._process_downwards(node_outputs[i-1][j], i, j)
                        vertical_inputs[i][j] = top_output
                    if j > 0:
                        left_output = self._process_rightwards(node_outputs[i][j-1], i, j, training=training)
                        lateral_inputs[i][j] = left_output

                    node_outputs[i][j] = top_output + left_output

            # Connect second half (Up-sampling streams) by iterating to the right, and upwards.
            for i in range(self.height - 1, -1, -1):
                for j in range(int(self.width / 2), self.width):
                    bottom_output, left_output = 0, 0
                    if i < self.height - 1:
                        bottom_output = self._process_upwards(node_outputs[i+1][j], i, j)
                        vertical_inputs[i][j] = bottom_output
                    if j > 0:
                        left_output = self._process_rightwards(node_outputs[i][j-1], i, j, training=training)
                        lateral_inputs[i][j] = left_output

                    node_outputs[i][j] = bottom_output + left_output

            # Final lateral connection.
            previous_output = node_outputs[0][self.width-1]
            final_output = self._process_rightwards(previous_output, 0, self.width, training=training)
            return final_output, node_outputs, lateral_inputs, vertical_inputs

    # Private helper functions.
    def _process_rightwards(self, input, i, j, training=False):

        # The input and output lateral connections should never be dropped, as they cutoff gradients hard.
        is_output = j == self.width
        force_alive = i == 0 and (j == 0 or j == self.width)
        total_dropout_rate = 0.0 if force_alive else self.connection_dropout_rate
        return LateralConnection(
            'right_%d%d' % (i, j),
            self.output_spec if is_output else self.lateral_specs[i],
            activation_fn=self.activation_fn,
            total_dropout_rate=total_dropout_rate,
            regularizer=self.regularizer
        ).get_forward(input, training=training)

    def _process_upwards(self, input, i, j):
        return UpSamplingConnection(
            'up_%d%d' % (i, j),
            self.upsample_specs[i],
            activation_fn=self.activation_fn,
            regularizer=self.regularizer
        ).get_forward(input)

    def _process_downwards(self, input, i, j):
        return DownSamplingConnection(
            'down_%d%d' % (i, j),
            self.downsample_specs[i],
            activation_fn=self.activation_fn,
            regularizer=self.regularizer
        ).get_forward(input)