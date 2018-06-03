import tensorflow as tf


def forward_warp(features, flow, t, max_image_area=1280*720):
    """
    See section 3 in https://arxiv.org/pdf/1711.05890.pdf.
    Note that the actual implementation here is not n^2, and should be linear in GPU memory.
    :param features: A Tensor. Features to be warped, of shape [batch_size, H, W, C].
    :param flow: A Tensor. Un-normalized flow in image pixel units, of shape [batch_size, H, W, 2].
    :param t: Float that specifies interpolation degree. 0 for not flowed, 1 for flow at full optical flow length.
    :param max_image_area: The maximum value for width * height of the input features. It is ok to specify a value
                           larger than the actual area (at the expense of performance), but it cannot be any less.
    """

    # Get target indices along with corresponding values to add.

    # Partition based on target index.

    # Aggregate for each index.

    # Scatter into output.


def get_pushed_pixels(features):
    """
    :param features: A Tensor. Of shape [batch_size, H, W, C].
    :return:
    """

