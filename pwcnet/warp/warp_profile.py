import numpy as np
import tensorflow as tf
import os
from pwcnet.warp.warp import backward_warp
from tensorflow.python.profiler.model_analyzer import Profiler, option_builder


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    height = 128
    width = 128
    im_channels = 128
    batch_size = 32
    num_runs = 10
    warmup_runs = 5

    # Create the graph.
    image_shape = [batch_size, height, width, im_channels]
    flow_shape = [batch_size, height, width, 2]
    image_placeholder = tf.placeholder(shape=image_shape, dtype=tf.float32)
    flow_placeholder = tf.placeholder(shape=flow_shape, dtype=tf.float32)
    warped = backward_warp(image_placeholder, flow_placeholder)
    grads = tf.gradients(warped, [image_placeholder, flow_placeholder])
    sess.run(tf.global_variables_initializer())

    # Create dummy images.
    image = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    flow = np.zeros(shape=[batch_size, height, width, 2], dtype=np.float32)
    image[:, 2:height - 2, 2:width - 2, :] = 1.0
    flow[:, 4:height - 4, 5:width - 5, :] = 1.0

    # Profiler data.
    run_metadata = tf.RunMetadata()
    folder = 'timelines'
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Profile both forward and backward passes together.
    profiler = Profiler(sess.graph)
    for i in range(num_runs + warmup_runs):
        _ = sess.run([warped, grads],
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata,
                     feed_dict={image_placeholder: image,
                                flow_placeholder: flow})
        if i >= warmup_runs:
            profiler.add_step(i - warmup_runs, run_metadata)

    # Print timing and memory.
    opts = (option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())
            .with_step(-1).build())
    profiler.profile_operations(options=opts)

    # Save timeline json.
    basename = 'backward-warp-timeline-code.json'
    save_path = os.path.join(folder, basename)
    opts = (option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())
            .with_step(-1)
            .with_timeline_output(save_path).build())

    profiler.profile_python(options=opts)

