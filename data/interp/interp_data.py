import glob
import multiprocessing
import os.path
import random
import numpy as np
from data.dataset import DataSet
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
        super().__init__(directory, batch_size, validation_size)

        # Initialized during load().
        self.train_dataset = None  # Tensorflow DataSet object.
        self.valid_dataset = None  # Tensorflow DataSet object.
        self.handle_placeholder = None  # Handle placeholder for switching between datasets.
        self.train_handle = None  # Handle to feed for the training dataset.
        self.validation_handle = None  # Handle to feed for the validation dataset.
        self.iterator = None  # Iterator for getting the next batch.
        self.train_iterator = None  # Iterator for getting just the train data.
        self.validation_iterator = None  # Iterator for getting just the validation data.
        self.next_sequences = None  # Data iterator batch.
        self.next_sequence_timing = None  # Data iterator batch.

        self.train_filename = 'interp_dataset_train.tfrecords'
        self.valid_filename = 'interp_dataset_valid.tfrecords'

        self.inbetween_locations = inbetween_locations

    def get_train_file_names(self):
        """
        Overridden.
        """
        return glob.glob(os.path.join(self.directory, '*' + self.train_filename))

    def get_validation_file_names(self):
        """
        :return: List of string.
        """
        return glob.glob(os.path.join(self.directory, '*' + self.valid_filename))

    def preprocess_raw(self, shard_size):
        """
        Overridden.
        """
        if self.verbose:
            print('Checking directory for data.')
        image_paths = self._get_data_paths()
        self._convert_to_tf_record(image_paths, shard_size)

    def load(self, session):
        """
        Overridden.
        """
        with tf.name_scope('dataset_ops'):

            train_datasets, valid_datasets = [], []
            for i in range(len(self.inbetween_locations)):
                inbetween_locations = self.inbetween_locations[i]
                train_dataset = self._load_dataset(self.get_train_file_names(), inbetween_locations)
                valid_dataset = self._load_dataset(self.get_validation_file_names(), inbetween_locations)
                if i == 0:
                    self.train_dataset = train_dataset
                    self.valid_dataset = valid_dataset
                else:
                    self.train_dataset.concatenate(train_dataset)
                    self.valid_dataset.concatenate(valid_dataset)

            self.train_dataset = self.train_dataset.shuffle(buffer_size=250).repeat()
            self.valid_dataset = self.valid_dataset.shuffle(buffer_size=250)

            self.handle_placeholder = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle_placeholder, self.train_dataset.output_types, self.train_dataset.output_shapes)
            self.next_sequences, self.next_sequence_timing = self.iterator.get_next()

            self.train_iterator = self.train_dataset.make_one_shot_iterator()
            self.validation_iterator = self.valid_dataset.make_initializable_iterator()

        self.train_handle = session.run(self.train_iterator.string_handle())
        self.validation_handle = session.run(self.validation_iterator.string_handle())

    def get_next_batch(self):
        """
        Overridden.
        """
        return self.next_sequences, self.next_sequence_timing

    def get_train_feed_dict(self):
        """
        Overridden.
        """
        return {self.handle_placeholder: self.train_handle}

    def get_validation_feed_dict(self):
        """
        Overridden.
        """
        return {self.handle_placeholder: self.validation_handle}

    def init_validation_data(self, session):
        """
        Overridden
        """
        session.run(self.validation_iterator.initializer)

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

        # Compute the validation start idx.
        # We might not satisfy self.validation_size as the split granularity will be at the shot level.
        if self.validation_size == 0:
            validation_seq_size = 0
        else:
            cur_items = 0
            validation_seq_size = 0
            for i in range(len(image_paths)):
                cur_items += len(image_paths[i])
                validation_seq_size = i + 1
                if cur_items >= self.validation_size:
                    break

        valid_start_idx = len(image_paths) - validation_seq_size
        _write(self.train_filename, range(0, valid_start_idx))
        _write(self.valid_filename, range(valid_start_idx, len(image_paths)))

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
                SHOT: tf.FixedLenFeature((), tf.string),
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            shot_len = tf.reshape(tf.cast(parsed_features[SHOT_LEN], tf.int32), ())
            H = tf.reshape(tf.cast(parsed_features[HEIGHT], tf.int32), ())
            W = tf.reshape(tf.cast(parsed_features[WIDTH], tf.int32), ())
            shot = tf.decode_raw(parsed_features[SHOT], tf.float32)
            shot = tf.reshape(shot, [shot_len, H, W, 3])

            # Decompose each shot into sequences of consecutive images.
            slice_locations = [1] + inbetween_locations + [1]
            return sliding_window_slice(shot, slice_locations)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)

        # Each element in the dataset is currently a group of sequences (grouped by video shot),
        # so we need to 'unbatch' them first.
        dataset = dataset.apply(tf.contrib.data.unbatch())

        # Add timing information.
        slice_indices = [1] + inbetween_locations + [1]
        slice_times = []
        for i in range(len(slice_indices)):
            if slice_indices[i] == 1:
                slice_times.append(i * 1.0 / (len(slice_indices) - 1))

        def add_timing(sequence):
            return sequence, slice_times

        dataset = dataset.map(add_timing)
        dataset = dataset.batch(self.batch_size)
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

    writer = tf.python_io.TFRecordWriter(os.path.join(directory, str(shard_id) + '_' + filename))
    for i in shard_range:

        # Read sequence from file.
        shot = [read_image(image_path, as_float=True) for image_path in image_paths[i]]
        shot = np.asarray(shot)
        H = shot[0].shape[0]
        W = shot[0].shape[1]

        # Write to tf record.
        shot_raw = shot.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    SHOT_LEN: tf_int64_feature(len(shot)),
                    HEIGHT: tf_int64_feature(H),
                    WIDTH: tf_int64_feature(W),
                    SHOT: tf_bytes_feature(shot_raw),
                }))
        writer.write(example.SerializeToString())
    writer.close()
