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
            for i in range(self.num_levels):
                print('Building level %d ...' % i)

        return images

    def _get_blur_kernel(self):
        """
        :return: A TF constant. Filter of shape [H, W].
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
        blur_kernel = np.outer(triangle[-1], triangle[-1])
        blur_kernel = blur_kernel / np.sum(blur_kernel)
        return tf.constant(blur_kernel)