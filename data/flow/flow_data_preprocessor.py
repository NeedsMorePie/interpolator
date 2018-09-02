import multiprocessing
import os.path
import random
import numpy as np
from joblib import Parallel, delayed
from common.utils.data import *
from common.utils.img import read_image
from common.utils.flow import read_flow_file
from data.flow.flow_data import FlowDataSet


class FlowDataPreprocessor:
    def __init__(self, directory, validation_size=1, max_flow=1000.0, shard_size=1, verbose=False):
        """
        :param directory: Str. Directory of the dataset file structure and tf records.
        :param validation_size: Int. Number of validation examples.
        :param data_source: Source of the data.
        :param max_flow: Float. Maximum flow magnitude of the flow image. Any examples with flow magnitude greater than
            this will be ignored.
        :param verbose: Bool.
        """
        self.directory = directory
        self.validation_size = validation_size
        self.max_flow = max_flow
        self.shard_size = shard_size
        self.verbose = verbose

    def get_data_paths(self):
        """
        :return: List of image_path strings, list of flow_path strings.
        """
        raise NotImplementedError('get_data_paths() is not implemented.')

    def preprocess_raw(self):
        image_a_paths, image_b_paths, flow_paths = self.get_data_paths()
        self._convert_to_tf_record(image_a_paths, image_b_paths, flow_paths, self.shard_size)

    def _convert_to_tf_record(self, image_a_paths, image_b_paths, flow_paths, shard_size):
        """
        :param image_paths: List of image_path strings.
        :param flow_paths: List of flow_np_path strings.
        :param shard_size: Maximum number of examples in each shard.
        :return: Nothing.
        """
        assert len(image_a_paths) == len(flow_paths)
        assert len(image_b_paths) == len(flow_paths)

        # Shuffle in unison.
        zipped = list(zip(image_a_paths, image_b_paths, flow_paths))
        random.shuffle(zipped)
        image_a_paths, image_b_paths, flow_paths = zip(*zipped)

        def _write(filename, iter_range):
            if self.verbose:
                print('Writing', len(iter_range),'data examples to the', filename, 'dataset.')

            sharded_iter_ranges = create_shard_ranges(iter_range, shard_size)

            Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading")(
                delayed(_write_shard)(shard_id, shard_range, image_a_paths, image_b_paths,
                                      flow_paths, filename, self.directory, self.verbose, self.max_flow)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        valid_start_idx = len(image_a_paths) - self.validation_size
        _write(FlowDataSet.TRAIN_FILENAME, range(0, valid_start_idx))
        _write(FlowDataSet.VALID_FILENAME, range(valid_start_idx, len(image_a_paths)))


def _write_shard(shard_id, shard_range, image_a_paths, image_b_paths, flow_paths, filename, directory, verbose,
                 max_flow):
    """
    :param shard_id: Index of the shard.
    :param shard_range: Iteration range of the shard.
    :param image_paths: Path of all images.
    :param flow_paths: Path of all flows.
    :param filename: Base name of the output shard.
    :param directory: Output directory.
    :param verbose: Whether to print to console.
    :param max_flow: Float. Maximum flow magnitude of the flow image. Any examples with flow magnitude greater than this
        will be ignored.
    :return: Nothing.
    """
    if verbose and len(shard_range) > 0:
        print('Writing to shard', shard_id, 'data points', shard_range[0], 'to', shard_range[-1])

    record_name = os.path.join(directory, str(shard_id) + '_' + filename)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(record_name, options=options)
    num_examples_written = 0
    for i in shard_range:
        # Read from file.
        flow = read_flow_file(flow_paths[i])
        if np.amax(np.linalg.norm(flow, axis=-1)) > max_flow:
            if verbose:
                print(flow_paths[i], 'has a flow magnitude greater than', max_flow)
            continue
        # Read and decode images as bytes to save memory.
        image_a = read_image(image_a_paths[i], as_float=False)
        image_b = read_image(image_b_paths[i], as_float=False)

        # Write to tf record.
        H = image_a.shape[0]
        W = image_a.shape[1]
        image_a_raw = image_a.tostring()
        image_b_raw = image_b.tostring()
        flow_raw = flow.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    FlowDataSet.HEIGHT: tf_int64_feature(H),
                    FlowDataSet.WIDTH: tf_int64_feature(W),
                    FlowDataSet.IMAGE_A_RAW: tf_bytes_feature(image_a_raw),
                    FlowDataSet.IMAGE_B_RAW: tf_bytes_feature(image_b_raw),
                    FlowDataSet.FLOW_RAW: tf_bytes_feature(flow_raw)
                }))
        writer.write(example.SerializeToString())
        num_examples_written += 1
    writer.close()

    if num_examples_written == 0:
        # Delete the file if nothing was written to it.
        if verbose:
            print(record_name, 'is empty')
        silently_remove_file(record_name)
