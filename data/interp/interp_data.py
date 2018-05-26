import glob
import multiprocessing
import os.path
import random
import numpy as np
from data.dataset import DataSet
from joblib import Parallel, delayed
from utils.data import *
from utils.img import read_image

SEQUENCE_LEN = 'sequence_len'
HEIGHT = 'height'
WIDTH = 'width'
SEQUENCE = 'sequence'
FLOW_RAW = 'flow_raw'


class InterpDataSet(DataSet):
    def __init__(self, directory, batch_size=1, validation_size=1):
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
        self.next_images_a = None  # Data iterator batch.
        self.next_images_b = None  # Data iterator batch.
        self.next_flows = None  # Data iterator batch.

        self.train_filename = 'flowdataset_train.tfrecords'
        self.valid_filename = 'flowdataset_valid.tfrecords'

        self.sequence_len = 3

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
            self.train_dataset = self._load_dataset(self.get_train_file_names(), True)
            self.valid_dataset = self._load_dataset(self.get_validation_file_names(), False)

            self.handle_placeholder = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle_placeholder, self.train_dataset.output_types, self.train_dataset.output_shapes)
            self.next_images_a, self.next_images_b, self.next_flows = self.iterator.get_next()

            self.train_iterator = self.train_dataset.make_one_shot_iterator()
            self.validation_iterator = self.valid_dataset.make_initializable_iterator()

        self.train_handle = session.run(self.train_iterator.string_handle())
        self.validation_handle = session.run(self.validation_iterator.string_handle())

    def get_next_batch(self):
        """
        Overridden.
        """
        return self.next_images_a, self.next_images_b, self.next_flows

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
                delayed(_write_shard)(shard_id, shard_range, image_paths, self.sequence_len,
                                      filename, self.directory, self.verbose)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        # Compute the validation start idx.
        # We might not satisfy self.validation_size as the split granularity will be at the shot level.
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

    def _load_dataset(self, filenames, repeat):
        """
        :param filenames: List of strings.
        :param repeat: Whether to repeat the dataset indefinitely.
        :return: Tensorflow dataset object.
        """
        def _parse_function(example_proto):
            features = {
                HEIGHT: tf.FixedLenFeature((), tf.int64, default_value=0),
                WIDTH: tf.FixedLenFeature((), tf.int64, default_value=0),
                IMAGE_A_RAW: tf.FixedLenFeature((), tf.string),
                IMAGE_B_RAW: tf.FixedLenFeature((), tf.string),
                FLOW_RAW: tf.FixedLenFeature((), tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            H = tf.reshape(tf.cast(parsed_features[HEIGHT], tf.int32), ())
            W = tf.reshape(tf.cast(parsed_features[WIDTH], tf.int32), ())
            image_a = tf.decode_raw(parsed_features[IMAGE_A_RAW], tf.float32)
            image_a = tf.reshape(image_a, [H, W, 3])
            image_b = tf.decode_raw(parsed_features[IMAGE_B_RAW], tf.float32)
            image_b = tf.reshape(image_b, [H, W, 3])
            flow = tf.decode_raw(parsed_features[FLOW_RAW], tf.float32)
            flow = tf.reshape(flow, [H, W, 2])
            return image_a, image_b, flow

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=250)
        dataset = dataset.batch(self.batch_size)
        if repeat:
            dataset = dataset.repeat()
        return dataset


def _write_shard(shard_id, shard_range, image_paths, sequence_len, filename, directory, verbose):
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
        images = [read_image(image_path, as_float=True) for image_path in image_paths[i]]

        for j in range(len(images) - sequence_len + 1):

            # Write to tf record.
            sequence = np.asarray([images[k] for k in range(j, j + sequence_len)])
            H = images[0].shape[0]
            W = images[0].shape[1]
            sequence_raw = sequence.to_string()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        SEQUENCE_LEN: tf_int64_feature(sequence_len),
                        HEIGHT: tf_int64_feature(H),
                        WIDTH: tf_int64_feature(W),
                        SEQUENCE: tf_bytes_feature(sequence_raw),
                    }))
            writer.write(example.SerializeToString())
    writer.close()