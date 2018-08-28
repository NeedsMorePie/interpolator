import numpy as np
import tensorflow as tf
from common.utils.profile import run_profiler
from pwcnet.warp.warp import backward_warp

if __name__ == '__main__':
    height = 128
    width = 128
    im_channels = 128
    batch_size = 32

    # Create the graph.
    image_shape = [batch_size, height, width, im_channels]
    flow_shape = [batch_size, height, width, 2]
    image_placeholder = tf.placeholder(shape=image_shape, dtype=tf.float32)
    flow_placeholder = tf.placeholder(shape=flow_shape, dtype=tf.float32)
    warped = backward_warp(image_placeholder, flow_placeholder)
    grads = tf.gradients(warped, [image_placeholder, flow_placeholder])

    # Create dummy images.
    image = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    flow = np.zeros(shape=[batch_size, height, width, 2], dtype=np.float32)
    image[:, 2:height - 2, 2:width - 2, :] = 1.0
    flow[:, 4:height - 4, 5:width - 5, :] = 1.0

    query = [warped, grads]
    feed_dict = {image_placeholder: image,
                 flow_placeholder: flow}
    run_profiler(query, feed_dict, name='backward-warp')
