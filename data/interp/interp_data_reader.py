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


class InterpDataSetReader:
    def __init__(self, directory, inbetween_locations, tf_record_name, batch_size=1):
        """
        :param inbetween_locations: A list of lists. Each element specifies where inbetweens will be placed,
                                    and each configuration will appear with uniform probability.
                                    For example, let a single element in the list be [0, 1, 0].
                                    With this, dataset elements will be sequences of 3 ordered frames,
                                    where the middle (inbetween) frame is 2 frames away from the first and last frames.
                                    The number of 1s must be the same for each list in this argument.
        """

        # Initialized during load().
        self.dataset = None  # Tensorflow DataSet object.
        self.handle_placeholder = None  # Handle placeholder for switching between datasets.
        self.handle = None  # Handle to feed for the dataset.
        self.iterator = None  # Iterator for getting the next batch.
        self.iterator = None  # Iterator for getting data.
        self.next_sequences = None  # Data iterator batch.
        self.next_sequence_timing = None  # Data iterator batch.

        self.batch_size = batch_size
        self.directory = directory
        self.tf_record_name = tf_record_name
        self.inbetween_locations = inbetween_locations

        # Check for number of ones, as the number of elements per-sequence must be the same.
        num_ones = (np.asarray(self.inbetween_locations[0]) == 1).sum()
        for i in range(1, len(self.inbetween_locations)):
            if (np.asarray(self.inbetween_locations[i]) == 1).sum() != num_ones:
                raise ValueError('The number of ones for each element in inbetween_locations must be the same.')

    def get_tf_record_names(self):
        return glob.glob(os.path.join(self.directory, '*' + self.tf_record_name))

    def load(self, session, repeat=False, shuffle=False):
        """
        Overridden.
        """
        with tf.name_scope(self.tf_record_name + '_dataset_ops'):
            for i in range(len(self.inbetween_locations)):
                inbetween_locations = self.inbetween_locations[i]
                dataset = self._load_dataset(self.get_tf_record_names(), inbetween_locations)
                if i == 0:
                    self.dataset = dataset
                else:
                    self.dataset = self.dataset.concatenate(dataset)

            buffer_size = 250
            if shuffle and repeat:
                self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size))
            elif shuffle:
                self.dataset = self.dataset.shuffle(buffer_size=buffer_size)
            elif repeat:
                self.dataset = self.dataset.repeat()

            self.handle_placeholder = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle_placeholder, self.dataset.output_types, self.dataset.output_shapes)
            self.next_sequences, self.next_sequence_timing = self.iterator.get_next()

            self.iterator = self.dataset.make_one_shot_iterator()

        self.handle = session.run(self.iterator.string_handle())

    def get_next_batch(self):
        """
        Overridden.
        """
        return self.next_sequences, self.next_sequence_timing

    def get_feed_dict(self):
        """
        Overridden.
        """
        return {self.handle_placeholder: self.handle}

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