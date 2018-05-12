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

        # Construct specs for lateral connections.
        # Entry specs[i][j] is the spec for the jth convolution for any connection in the ith row.
        self.lateral_specs = []
        self.upsample_specs = []
        self.downsample_specs = []
        for i in range(self.height):
            row_lateral_specs, row_upsample_specs, row_downsample_specs = [], [], []
            for j in range(self.num_lateral_convs):
                row_lateral_specs.append([32, 1])

            # Row 0 actually does not apply for down-sampling connections.
            if i > 0:
                row_downsample_specs.append()

            # Row n-1 actually does not apply for up-sampling connections.
            if i < self.height - 1:
                row_upsample_specs.append()

            self.lateral_specs.append()




        # Construct spec



    def get_forward(self, features):
        """
        :param features: A Tensor. Input feature maps of shape [batch_size, H, W, num_features]
        :return: A Tensor. Will take on the same shape as param features.
        """

        # First lateral connection.
        previous_output = LateralConnection('lateral_in', )
        grid_width = self.width + 1
        grid_hegith = self.height + 1

        # Connect first half by iterating to the right, and downwards.


        # Connect second half by iterating to the right, and upwards.



        # Final lateral connection.

