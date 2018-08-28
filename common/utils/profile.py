import os
import tensorflow as tf
import time
from tensorflow.python.profiler.model_analyzer import Profiler, option_builder


PROFILE_CMDS = ['code', 'scope', 'graph']


def run_profiler(query, feed_dict, num_runs=10, warmup_runs=5, name='profile'):
    """
    Runs the Tensorflow profiler on a given query and feed_dict.
    Timelines are placed under timelines/<name>.
    :param query: List of Tensors.
    :param feed_dict: Dict.
    :param num_runs: Int. Number of runs to profile for.
    :param warmup_runs: Int. Number of warm-up runs before num_runs.
    :param name: Str. Output file name.
    :return: Nothing.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Profiler data.
    run_metadata = tf.RunMetadata()
    folder = os.path.join('timelines', name)
    os.makedirs(folder, exist_ok=True)

    profiler = Profiler(sess.graph)
    for i in range(num_runs + warmup_runs):
        sess.run(query, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata, feed_dict=feed_dict)
        if i >= warmup_runs:
            profiler.add_step(i - warmup_runs, run_metadata)

    # Print timing and memory.
    opts = (option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())
            .with_step(-1).build())
    profiler.profile_operations(options=opts)

    # Save timeline json.
    save_path = os.path.join(folder, 'timeline.json')
    opts = (option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())
            .with_step(-1)
            .with_timeline_output(save_path).build())
    profiler.profile_name_scope(options=opts)

    sess.close()


def run_tensorboard_profiler(query, feed_dict, num_runs=10, warmup_runs=5, name='profile'):
    """
    Runs the Tensorboard profiler on a given query and feed_dict.
    Timelines and Tensorboard visualizations are placed under timelines/<name>.
    :param query: List of Tensors.
    :param feed_dict: Dict.
    :param num_runs: Int. Number of runs to time.
    :param warmup_runs: Int. Number of warm-up runs before num_runs.
    :param name: Str. Output file name.
    :return: Nothing.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Profiler data.
    run_metadata = tf.RunMetadata()
    folder = os.path.join('timelines', name)
    os.makedirs(folder, exist_ok=True)
    train_writer = tf.summary.FileWriter(folder, sess.graph)

    for i in range(warmup_runs):
        sess.run(query, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata, feed_dict=feed_dict)
    sess.run(query, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
             run_metadata=run_metadata, feed_dict=feed_dict)
    save_timeline(run_metadata, folder)

    # Do a simple timing.
    total_time = 0.0
    for i in range(num_runs):
        start = time.time()
        sess.run(query, feed_dict=feed_dict)
        end = time.time()
        total_time += end - start
    total_time /= num_runs
    print('Average total runtime is:', total_time * 1000.0, 'ms.')

    # Write Tensorboard graph.
    train_writer.add_run_metadata(run_metadata, 'profile', global_step=0)
    train_writer.close()
    sess.close()


def save_timeline(run_metadata, directory, name=''):
    """
    :param run_metadata: Tensorflow run metadata.
    :param directory: Str. Directory to save into.
    :param name: Str. Name of the run.
    :return: Nothing.
    """
    for profile_cmd in PROFILE_CMDS:
        if name != '':
            file_name = name + '_timeline_' + profile_cmd + '.json'
        else:
            file_name = 'timeline_' + profile_cmd + '.json'
        out_path = os.path.join(directory, file_name)
        inference_profile_opts = (tf.profiler.ProfileOptionBuilder(
            tf.profiler.ProfileOptionBuilder.time_and_memory())
                                  .with_timeline_output(out_path).build())
        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=run_metadata,
            cmd=profile_cmd,
            options=inference_profile_opts
        )
