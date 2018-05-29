import glob
import multiprocessing
import os.path
import random
import numpy as np
from data.dataset import DataSet
from data.interp.interp_data_reader import InterpDataSetReader
from joblib import Parallel, delayed
from utils.data import *
from utils.img import read_image
from utils.misc import sliding_window_slice

SHOT_LEN = 'shot_len'
HEIGHT = 'height'
WIDTH = 'width'
SHOT = 'shot'


class InterpDataSet(DataSet):
    def __init__(self, directory, inbetween_locations, batch_size=1, validation_size=0):
        """
        :param inbetween_locations: A list of lists. Each element specifies where inbetweens will be placed,
                                    and each configuration will appear with uniform probability.
                                    For example, let a single element in the list be [0, 1, 0].
                                    With this, dataset elements will be sequences of 3 ordered frames,
                                    where the middle (inbetween) frame is 2 frames away from the first and last frames.
                                    The number of 1s must be the same for each list in this argument.
        """
        super().__init__(directory, batch_size, validation_size=validation_size)

        #out_dir = os.path.join(directory, 'tfrecords')
        out_dir = directory
        self.output_directory = out_dir

        self.train_tf_record_name = 'interp_dataset_train.tfrecords'
        self.validation_tf_record_name = 'interp_dataset_validation.tfrecords'
        self.train_data = InterpDataSetReader(out_dir, inbetween_locations,
                                              self.train_tf_record_name, batch_size=batch_size)
        self.validation_data = InterpDataSetReader(out_dir, inbetween_locations,
                                              self.validation_tf_record_name, batch_size=batch_size)

    def get_tf_record_names(self):
        """
        :return: All the tf record names that were saved with this instance.
        """
        return self.get_train_file_names() + self.get_validation_file_names()

    def get_train_file_names(self):
        """
        Overriden.
        """
        return self.train_data.get_tf_record_names()

    def get_validation_file_names(self):
        """
        Overriden.
        """
        return self.validation_data.get_tf_record_names()

    def preprocess_raw(self, shard_size):
        """
        Overridden.
        """
        if self.verbose:
            print('Checking directory for data.')
        image_paths = self._get_data_paths()
        self._convert_to_tf_record(image_paths, shard_size)

    def load(self, session, repeat=False, shuffle=False):
        """
        Overridden.
        """
        self.train_data.load(session, repeat=True, shuffle=True)
        self.validation_data.load(session, repeat=False, shuffle=False)

    def get_next_batch(self):
        """
        Overridden.
        """
        return self.train_data.get_next_batch()

    def get_feed_dict(self):
        """
        Overridden.
        """
        return self.train_data.get_feed_dict()

    def _get_data_paths(self):
        """
        Gets the paths of images from a directory that is organized with each video shot in its own folder.
        The image order for each sequence must be obtainable by sorting their names.
        :return: List of list of image names, where image_paths[0][0] is the first image in the first video shot.
        """
        image_names = []
        extensions = ['*.jpg']
        for item in os.listdir(self.directory):
            path = os.path.join(self.directory, item)
            if os.path.isdir(path):
                cur_names = []
                for ext in extensions:
                    cur_names += glob.glob(os.path.join(path, '**', ext), recursive=True)
                cur_names.sort()
                image_names.append(cur_names)
        return image_names

    def _convert_to_tf_record(self, image_paths, shard_size):
        """
        :param image_paths: List of list of image names,
                            where image_paths[0][0] is the first image in the first video shot.
        :return: Nothing.
        """
        random.shuffle(image_paths)
        def _write(filename, iter_range):
            if self.verbose:
                print('Writing', len(iter_range), 'data examples to the', filename, 'dataset.')

            sharded_iter_ranges = create_shard_ranges(iter_range, shard_size)

            Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading")(
                delayed(_write_shard)(shard_id, shard_range, image_paths,
                                      filename, self.directory, self.verbose)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        valid_start_idx = len(image_paths) - self.validation_size
        _write(self.train_tf_record_name, range(0, valid_start_idx))
        _write(self.validation_tf_record_name, range(valid_start_idx, len(image_paths)))

    def _load_dataset(self, filenames, inbetween_locations):
        """
        :param filenames: List of strings.
        :param inbetween_locations: An element of self.inbetween_locations.
        :return: Tensorflow dataset object.
        """
        def _parse_function(example_proto):
            features = {
                SHOT_LEN: tf.FixedLenFeature((), tf.int64, default_value=0),
                HEIGHT: tf.FixedLenFeature((), tf.int64, default_value=0),
                WIDTH: tf.FixedLenFeature((), tf.int64, default_value=0),
                SHOT: tf.VarLenFeature(tf.string),
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            shot_len = tf.reshape(tf.cast(parsed_features[SHOT_LEN], tf.int32), ())
            H = tf.reshape(tf.cast(parsed_features[HEIGHT], tf.int32), ())
            W = tf.reshape(tf.cast(parsed_features[WIDTH], tf.int32), ())

            shot_bytes = tf.sparse_tensor_to_dense(parsed_features[SHOT], default_value=tf.as_string(0))
            shot = tf.map_fn(lambda bytes: tf.image.decode_image(bytes), shot_bytes, dtype=(tf.uint8))
            shot = tf.image.convert_image_dtype(shot, tf.float32)
            shot = tf.reshape(shot, [shot_len, H, W, 3])

            # Decompose each shot into sequences of consecutive images.
            slice_locations = [1] + inbetween_locations + [1]
            return sliding_window_slice(shot, slice_locations)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_parallel_calls=multiprocessing.cpu_count())

        # Each element in the dataset is currently a group of sequences (grouped by video shot),
        # so we need to 'unbatch' them first.
        dataset = dataset.apply(tf.contrib.data.unbatch())

        # Add timing information.
        slice_indices = [1] + inbetween_locations + [1]
        slice_times = []
        for i in range(len(slice_indices)):
            if slice_indices[i] == 1:
                slice_times.append(i * 1.0 / (len(slice_indices) - 1))

        def _add_timing(sequence):
            return sequence, tf.constant(slice_times)

        dataset = dataset.apply(tf.contrib.data.map_and_batch(_add_timing, self.batch_size))
        dataset = dataset.prefetch(buffer_size=1)
        return dataset


def _write_shard(shard_id, shard_range, image_paths, filename, directory, verbose):
    """
    :param shard_id: Index of the shard.
    :param shard_range: Iteration range of the shard.
    :param image_paths: List of list of image names.
    :param filename: Base name of the output shard.
    :param directory: Output directory.
    :return: Nothing.
    """
    if verbose and len(shard_range) > 0:
        print('Writing to shard', shard_id, 'data points', shard_range[0], 'to', shard_range[-1])

    path = os.path.join(directory, str(shard_id) + '_' + filename)
    writer = tf.python_io.TFRecordWriter(path)
    for i in shard_range:
        if len(image_paths[i]) <= 0:
            continue

        # Read sequence from file.
        reference_image = read_image(image_paths[i][0], as_float=True)

        shot_raw = []
        for image_path in image_paths[i]:
            with open(image_path, 'rb') as fp:
                shot_raw.append(fp.read())

        H = reference_image.shape[0]
        W = reference_image.shape[1]

        # Write to tf record.
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    SHOT_LEN: tf_int64_feature(len(shot_raw)),
                    HEIGHT: tf_int64_feature(H),
                    WIDTH: tf_int64_feature(W),
                    SHOT: tf_bytes_list_feature(shot_raw),
                }))
        writer.write(example.SerializeToString())
    writer.close()
