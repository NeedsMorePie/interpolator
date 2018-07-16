import numpy as np
import tensorflow as tf
import time
import os
from context_interp.model import ContextInterp
from utils.tf import AdamaxOptimizer


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    height = 256
    width = 256
    im_channels = 3
    batch_size = 8
    num_runs = 30
    warmup_runs = 20

    # Create the graph.
    print('Creating the graph...')
    model = ContextInterp('context_interp_profile')
    image_a_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
    image_b_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
    gt_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
    output_tensors = model.get_forward(image_a_placeholder, image_b_placeholder, 0.5)
    interpolated_tensor, warped_a_b_tensor, warped_b_a_tensor, _, _ = output_tensors
    grad_tensors = tf.gradients(interpolated_tensor, [image_a_placeholder, image_b_placeholder])
    loss_tensor = model.get_training_loss(interpolated_tensor, gt_placeholder)
    train_op = AdamaxOptimizer(1E-4, beta1=0.9, beta2=0.999).minimize(loss_tensor)
    sess.run(tf.global_variables_initializer())

    # Create dummy images.
    image_a = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    image_b = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
    image_a[:, 2:height - 2, 2:width - 2, :] = 1.0
    image_b[:, 4:height - 4, 5:width - 5, :] = 1.0

    # Profiler data.
    run_metadata = tf.RunMetadata()
    profile_cmds = ['code', 'scope', 'graph']
    folder = 'timelines'
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Profile the forward pass.
    inference_avg = 0
    for i in range(num_runs + warmup_runs):
        t1 = time.time()
        run_metadata_feed = run_metadata if (i == num_runs + warmup_runs - 1) else None
        _ = sess.run(interpolated_tensor,
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata_feed,
                     feed_dict={image_a_placeholder: image_a,
                                image_b_placeholder: image_b,
                                gt_placeholder: np.zeros(shape=image_a.shape)}
                     )
        if i >= warmup_runs:
            dt = time.time() - t1
            print('Current inference time: %f' % dt)
            inference_avg += dt

    print('Averaged inference time: %f' % (inference_avg / num_runs))
    print('--------------------------------')
    for profile_cmd in profile_cmds:
        name = 'ctx-interp-inference-timeline-' + profile_cmd + '.json'
        path = os.path.join(folder, name)
        inference_profile_opts = (tf.profiler.ProfileOptionBuilder(
            tf.profiler.ProfileOptionBuilder.time_and_memory())
                                  .with_timeline_output(path).build())
        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=run_metadata,
            cmd=profile_cmd,
            options=inference_profile_opts
        )

    # Profile the training time.
    training_avg = 0
    for i in range(num_runs + warmup_runs):
        t1 = time.time()
        run_metadata_feed = run_metadata if (i == num_runs + warmup_runs - 1) else None
        _ = sess.run(train_op,
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata_feed,
                     feed_dict={image_a_placeholder: image_a,
                                image_b_placeholder: image_b,
                                gt_placeholder: np.zeros(shape=image_a.shape)})
        if i >= warmup_runs:
            dt = time.time() - t1
            print('Current training time: %f' % dt)
            training_avg += dt

    print('Averaged training time: %f' % (training_avg / num_runs))
    for profile_cmd in profile_cmds:
        name = 'ctx-interp-training-timeline-' + profile_cmd + '.json'
        path = os.path.join(folder, name)
        inference_profile_opts = (tf.profiler.ProfileOptionBuilder(
            tf.profiler.ProfileOptionBuilder.time_and_memory())
                                  .with_timeline_output(path).build())
        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=run_metadata,
            cmd=profile_cmd,
            options=inference_profile_opts
        )
