import numpy as np
import tensorflow as tf
from pwcnet.model import PWCNet
from utils.profile import run_tensorboard_profiler


if __name__ == '__main__':
    height = 384
    width = 448
    im_channels = 3
    batch_size = 8

    # Create the graph.
    image_shape = [batch_size, height, width, im_channels]
    image_a_placeholder = tf.placeholder(shape=image_shape, dtype=tf.float32)
    image_b_placeholder = tf.placeholder(shape=image_shape, dtype=tf.float32)
    pwcnet = PWCNet()
    flow, _ = pwcnet.get_forward(image_a_placeholder, image_b_placeholder)
    grads = tf.gradients(flow, [image_a_placeholder, image_b_placeholder])

    # Create dummy images.
    image_a = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    image_b = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    image_a[:, 2:height - 2, 2:width - 2, :] = 1.0
    image_b[:, 4:height - 4, 5:width - 5, :] = 1.0

    query = [flow, grads]
    feed_dict = {image_a_placeholder: image_a,
                 image_b_placeholder: image_b}

    run_tensorboard_profiler(query, feed_dict, num_runs=5, warmup_runs=3, name='pwc-net')
