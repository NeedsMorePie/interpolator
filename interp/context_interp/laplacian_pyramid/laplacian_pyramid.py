import tensorflow as tf
import numpy as np


# References:
# http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb#laplacian
# http://www.cse.psu.edu/~rtc12/CSE486/lecture10.pdf
# https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrup#pyrup
class LaplacianPyramid:
    def __init__(self, num_levels, name='laplacian_pyramid', filter_side_len=5):
        """
        :param num_levels: The number of pyramid levels.
                           At each level the image width and height are 2x lower than the previous one.
        :param name: Str. For name scoping.
        :param filter_side_len: The width and height of the approximate gaussian blur filter.
        """
        self.name = name
        self.num_levels = num_levels
        self.filter_side_len = filter_side_len

    def get_forward(self, images):
        """
        :param images: A Tensor. Images of shape [batch_size, H, W, C].
        :return: laplacian_levels: A list of Tensors of length self.num_levels.
                                   Each level (from 0) has tensors of shape [batch_size, H / 2^level, W / 2^level, C].
                 gaussian_levels: Same format as laplacian_levels.
                 reconstructed: A Tensor. Input reconstruction of shape [batch_size, H, W, C].
        """
        with tf.name_scope(self.name):
            image_height = tf.shape(images)[1]
            image_width = tf.shape(images)[2]
            num_channels = tf.shape(images)[-1]

            # Make sure that we are at integer values.
            final_height = image_height / 2 ** (self.num_levels - 1)
            final_width = image_width / 2 ** (self.num_levels - 1)
            height_check = tf.Assert(tf.equal(final_height, tf.floor(final_height)), [final_height])
            width_check = tf.Assert(tf.equal(final_width, tf.floor(final_width)), [final_width])

            with tf.control_dependencies([height_check, width_check]):

                # Build the gaussian pyramid.
                with tf.name_scope('gaussian_pyramid'):
                    gaussian_levels = [images]
                    blur_filter = self._get_blur_filter(num_channels)
                    for i in range(1, self.num_levels):
                        downsampled = self._pyr_down(gaussian_levels[i - 1], blur_filter)
                        gaussian_levels.append(downsampled)

                # Build the laplacian pyramid.
                with tf.name_scope('laplacian_pyramid'):
                    laplacian_levels = []
                    for i in range(len(gaussian_levels) - 1):
                        upsampled = self._pyr_up(gaussian_levels[i + 1], blur_filter)
                        diff = gaussian_levels[i] - upsampled
                        laplacian_levels.append(diff)
                    laplacian_levels.append(gaussian_levels[-1])

                # Build the reconstruction.
                with tf.name_scope('reconstruction'):
                    reconstructed = gaussian_levels[-1]
                    for i in range(self.num_levels - 2, -1, -1):
                        reconstructed = self._pyr_up(reconstructed, blur_filter)
                        reconstructed += laplacian_levels[i]

        return laplacian_levels, gaussian_levels, reconstructed

    def _pyr_down(self, images, filter):
        """
        :param images: A Tensor. Images of shape [batch_size, H, W, C].
        :param filter: A Tensor. Convolution filter of shape [H, W, in_channels, out_channels].
        :return: A Tensor. Images of shape [batch_size, H/2, W/2, C].
        """
        return tf.nn.conv2d(images, filter, [1, 2, 2, 1], 'SAME')

    def _pyr_up(self, images, filter):
        """
        :param images: A Tensor. Images of shape [batch_size, H, W, C].
        :param filter: A Tensor. Convolution filter of shape [H, W, in_channels, out_channels].
        :return: A Tensor. Images of shape [batch_size, H*2, W*2, C].
        """
        batch_size = tf.shape(images)[0]
        H, W, C = tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]
        upsampled = tf.nn.conv2d_transpose(images, 4 * filter, (batch_size, 2 * H, 2 * W, C), [1, 2, 2, 1])
        return upsampled

    def _get_blur_filter(self, num_in_channels):
        """
        :param num_in_channels: The number of input channels to the convolutions.
        :return: A TF constant. Filter of shape [H, W, in_channels, in_channels].
                 Note that H = W = self.filter_side_len.
                 The filter will effectively be depth-wise (1-to-1 mapping from input to output channels).
        """

        # Generate Pascal's triangle.
        # The final row of the triangle is an integer approximation of the normal distribution.
        # See: http://www.cse.psu.edu/~rtc12/CSE486/lecture10.pdf for more details.
        triangle = [[1, 1]]
        for i in range(1, self.filter_side_len - 1):
            cur_row = [1]
            prev_row = triangle[i - 1]
            for j in range(len(prev_row) - 1):
                cur_row.append(prev_row[j] + prev_row[j + 1])
            cur_row.append(1)
            triangle.append(cur_row)

        # Get the TF kernel variable.
        blur_filter_np = np.outer(triangle[-1], triangle[-1])
        blur_filter_np = blur_filter_np / np.sum(blur_filter_np)
        blur_filter = [[tf.constant(blur_filter_np, dtype=tf.float32)]]
        blur_filter = tf.tile(blur_filter, [num_in_channels, num_in_channels, 1, 1])
        blur_filter = tf.transpose(blur_filter, [2, 3, 0, 1])
        blur_filter = blur_filter * tf.eye(num_in_channels)
        return blur_filter