import tensorflow as tf
from gridnet.connections import UpSamplingConnection, DownSamplingConnection, LateralConnection


class GridNet:
    def __init__(self, height, width, row_channel_sizes,
                 name='gridnet',
                 num_lateral_convs=2,
                 num_upsample_convs=2,
                 num_downsample_convs=2,
                 use_batch_norm=False,
                 connection_dropout_rate=0.0,
                 regularizer=None):
        """
        See https://arxiv.org/pdf/1707.07958.pdf, and modifications made in https://arxiv.org/pdf/1803.10967.pdf.
        :param height: Height of the GridNet.
        :param width: Width of the GridNet. Must be an even number to ensure symmetry.
        :param row_channel_sizes: List of channel sizes for rows. Must have length equal to height.
        :param name: Str. For variable scoping.
        :param num_lateral_convs: Number of convolutions in each lateral connection.
        :param num_upsample_convs: Number of convolutions in each up-sampling connection.
        :param num_downsample_convs: Number of convolutions in each down-sampling connection.
        :param use_batch_norm: Whether to use batch normalization.
        :param connection_dropout_rate: E.g if 0.5, drops out each connection (not individual neurons) with 50% chance.
        """

        if height <= 0:
            raise ValueError('Height must be >= 1.')

        if height != len(row_channel_sizes):
            raise ValueError('Height must match with length of row_channel_sizes.')

        if width % 2 != 0:
            raise ValueError('Width must be an even number, to enforce GridNet symmetry.')

        if width <= 0:
            raise ValueError('Width must be non-zero.')

        self.width = width
        self.height = height
        self.row_channel_sizes = row_channel_sizes
        self.name = name
        self.num_lateral_convs = num_lateral_convs
        self.num_downsample_convs = num_downsample_convs
        self.num_upsample_convs = num_upsample_convs
        self.use_batch_norm = use_batch_norm
        self.connection_dropout_rate = connection_dropout_rate

        # More settings
        self.activation_fn = tf.keras.layers.PReLU

        # Construct specs for lateral connections.
        # Entry specs[i][j] is the spec for the jth convolution for any connection in the ith row.
        self.lateral_specs = []
        self.upsample_specs = []
        self.downsample_specs = []
        for i in range(self.height):
            row_lateral_specs, row_upsample_specs, row_downsample_specs = [], [], []
            num_filters = row_channel_sizes[i]
            common_spec = [num_filters, 1]
            row_lateral_specs = [common_spec for j in range(self.num_lateral_convs)]

            if i > 0:
                row_downsample_specs = [common_spec for j in range(self.num_downsample_convs)]

            if i < self.height - 1:
                row_upsample_specs = [common_spec for j in range(self.num_upsample_convs)]

            self.lateral_specs.append(row_lateral_specs)
            self.upsample_specs.append(row_upsample_specs)
            self.downsample_specs.append(row_downsample_specs)


    def get_forward(self, features, reuse_variables=False):
        """
        :param features: A Tensor. Input feature maps of shape [batch_size, H, W, num_features]
        :return: final_output: A Tensor. Will take on the same shape as param features.
                 grid_outputs: A 2D list of Tensors. Represents the output at each grid node.
        """

        with tf.variable_scope(self.name, reuse=reuse_variables):
            grid_outputs = [[None for x in range(self.width)] for y in range(self.height)]

            # First lateral connection.
            grid_outputs[0][0] = self.process_rightwards(features, 0, 0)

            # Connect first half by iterating to the right, and downwards.
            for i in range(self.height):
                for j in range(self.width / 2):
                    if i == 0 and j == 0:
                        continue

                    top_output, left_output = 0, 0
                    if i > 0:
                        top_output = self.process_downwards(grid_outputs[i-1][j], i, j)
                    if j > 0:
                        left_output = self.process_rightwards(grid_outputs[i][j-1], i, j)

                    grid_outputs[i][j] = top_output + left_output

            # Connect second half by iterating to the right, and upwards.
            for i in range(self.height - 1, 0, -1):
                for j in range(self.width / 2, self.width):
                    bottom_output, left_output = 0, 0
                    if i < self.height - 1:
                        bottom_output = self.process_upwards(grid_outputs[i+1][j], i, j)
                    if j > 0:
                        left_output = self.process_rightwards(grid_outputs[i][j-1], i, j)

                    grid_outputs[i][j] = bottom_output + left_output

            final_output = grid_outputs[self.height-1][self.width-1]
            return final_output, grid_outputs

    # Helper functions.
    def process_rightwards(self, input, i, j):
        return LateralConnection(
            'right_%d%d' % (i, j),
            self.lateral_specs[i],
            activation_fn=self.activation_fn,
            use_batch_norm=self.use_batch_norm,
            total_dropout_rate=self.connection_dropout_rate
        ).get_forward(input)

    def process_upwards(self, input, i, j):
        return UpSamplingConnection(
            'up_%d%d' % (i, j),
            self.upsample_specs[i],
            self.activation_fn,
            activation_fn=self.activation_fn,
            use_batch_norm=self.use_batch_norm,
            total_dropout_rate=self.connection_dropout_rate
        ).get_forward(input)

    def process_downwards(self, input, i, j):
        return DownSamplingConnection(
            'down_%d%d' % (i, j),
            self.downsample_specs[i],
            self.activation_fn,
            activation_fn=self.activation_fn,
            use_batch_norm=self.use_batch_norm,
            total_dropout_rate=self.connection_dropout_rate
        ).get_forward(input)