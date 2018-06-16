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
        :return: A Tensor. Images at all the pyramid levels, of shape [batch_size, levels, H, W, C].
        """
        with tf.variable_scope(self.name):
            num_channels = tf.shape(images)[-1]
            levels = [images]
            blur_filter = self._get_blur_filter(num_channels)
            for i in range(1, self.num_levels):
                print('Building level %d ...' % i)
                blurred = tf.nn.depthwise_conv2d(levels[i - 1], blur_filter, 4 * [1], 'VALID')
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
        blur_filter = [tf.constant(blur_filter_np)]
        blur_filter = tf.tile(blur_filter, [num_in_channels, 1, 1])
        blur_filter = tf.transpose(blur_filter, [1, 2, 0])
        blur_filter = tf.expand_dims(blur_filter, axis=-1)
        return blur_filter