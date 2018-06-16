import tensorflow as tf
import numpy as np


# References:
# http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb#laplacian
# http://www.cse.psu.edu/~rtc12/CSE486/lecture10.pdf
class ImagePyramid:

    def __init__(self, num_levels, name='image_pyr', filter_side_len=5):
        """
        :param num_levels: The number of pyramid levels.
                       At each level the image width and height are 2x lower than the previous one.
        :param name: Str. For variable scoping.
        :param filter_side_len: The width and height of the approximate gaussian blur filter.
        """
        self.name = name
        self.num_levels = num_levels
        self.filter_side_len = filter_side_len

    def get_forward(self, images):
        """
        :param images: A Tensor. Images of shape [batch_size, H, W, C].
        :return: A list of Tensors of length self.num_levels.
                 Each level (from 0) has tensors of shape [batch_size, H / 2^level, W / 2^level, C].
        """
        with tf.variable_scope(self.name):
            image_height = tf.shape(images)[1]
            image_width = tf.shape(images)[2]
            num_channels = tf.shape(images)[-1]

            # Make sure that we are at integer values.
            final_height = image_height / 2 ** (self.num_levels - 1)
            final_width = image_width / 2 ** (self.num_levels - 1)
            height_check = tf.Assert(tf.equal(final_height, tf.floor(final_height)), [final_height])
            width_check = tf.Assert(tf.equal(final_width, tf.floor(final_width)), [final_width])

            # Build the pyramid by depth-wise gaussian blurring followed by 2x down-sampling.
            with tf.control_dependencies([height_check, width_check]):
                levels = [images]
                blur_filter = self._get_blur_filter(num_channels)
                for i in range(1, self.num_levels):
                    blurred = tf.nn.depthwise_conv2d(levels[i - 1], blur_filter, 4 * [1], 'VALID')
                    new_height = tf.cast(image_height / 2 ** i, tf.int32)
                    new_width = tf.cast(image_width / 2 ** i, tf.int32)
                    blurred = tf.image.resize_bilinear(blurred, [new_height, new_width])
                    levels.append(blurred)
        return levels

    def _get_blur_filter(self, num_in_channels):
        """
        :param num_in_channels: The number of input channels to the convolutions.
        :return: A TF constant. Filter of shape [H, W, in_channels, 1].
                 Note that H = W = self.filter_side_len.
        """

        # Generate Pascal's triangle.
        triangle = [[1, 1]]
        for i in range(1, self.filter_side_len-1):
            cur_row = [1]
            prev_row = triangle[i - 1]
            for j in range(len(prev_row) - 1):
                cur_row.append(prev_row[j] + prev_row[j+1])
            cur_row.append(1)
            triangle.append(cur_row)

        # Get the TF kernel variable.
        blur_filter_np = np.outer(triangle[-1], triangle[-1])
        blur_filter_np = blur_filter_np / np.sum(blur_filter_np)
        blur_filter = [tf.constant(blur_filter_np, dtype=tf.float32)]
        blur_filter = tf.tile(blur_filter, [num_in_channels, 1, 1])
        blur_filter = tf.transpose(blur_filter, [1, 2, 0])
        blur_filter = tf.expand_dims(blur_filter, axis=-1)
        return blur_filter