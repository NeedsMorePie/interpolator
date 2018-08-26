import numpy as np
import tensorflow as tf
from pwcnet.cost_volume.cost_volume import cost_volume
from utils.profile import run_profiler


if __name__ == '__main__':
    height = 128
    width = 128
    im_channels = 128
    batch_size = 32
    search_range = 4

    # Create the graph.
    image_shape = [batch_size, height, width, im_channels]
    image_a_placeholder = tf.placeholder(shape=image_shape, dtype=tf.float32)
    image_b_placeholder = tf.placeholder(shape=image_shape, dtype=tf.float32)
    cv = cost_volume(image_a_placeholder, image_b_placeholder, search_range=search_range)
    grads = tf.gradients(cv, [image_a_placeholder, image_b_placeholder])

    # Create dummy images.
    image_a = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    image_b = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    image_a[:, 2:height - 2, 2:width - 2, :] = 1.0
    image_b[:, 4:height - 4, 5:width - 5, :] = 1.0

    query = [cv, grads]
    feed_dict = {image_a_placeholder: image_a,
                 image_b_placeholder: image_b}

    run_profiler(query, feed_dict, name='cost-volume')
