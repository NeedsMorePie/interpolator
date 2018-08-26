import glob
import multiprocessing
import os.path
import numpy as np
from utils.data import *
from utils.tf import sliding_window_slice, tf_coin_flip

SHOT_LEN = 'shot_len'
WIDTH = 'width'
HEIGHT = 'height'
SHOT = 'shot'


class InterpDataSetReader:
    def __init__(self, directory, inbetween_locations, tf_record_name, batch_size=1,
                 do_augment=False,
                 crop_size=(256, 256)):
        """
        :param inbetween_locations: A list of lists. Each element specifies where inbetweens will be placed,
                                    and each configuration will appear with uniform probability.

                                    For example, Let frame0 be the start of a sequence. Then:
                                        [1] equates to [frame0, frame1, frame2]
                                        [0, 1, 0] equates to [frame0, frame2, frame4]
                                        [1, 0, 0] equates to [frame0, frame1, frame4]
        :param crop_size: Tuple of (int (H), int (W)). Size to crop the training examples to before feeding to network.
                           If None, then no cropping will be performed.
        :param do_augment: Whether to augment the data when it's read in.
                           If False, the image crop will always be taken in the center.
        """

        # Initialized during load().
        self.dataset = None  # Tensorflow DataSet object.
        self.handle = None  # Handle to feed for the dataset.
        self.iterator = None  # Iterator for getting the next batch.

        self.batch_size = batch_size
        self.crop_size = crop_size
        self.do_augment = do_augment
        self.directory = directory
        self.tf_record_name = tf_record_name
        self.inbetween_locations = inbetween_locations

        # Check for number of ones, as the number of elements per-sequence must be the same.
        num_ones = (np.asarray(self.inbetween_locations[0]) == 1).sum()
        for i in range(1, len(self.inbetween_locations)):
            if (np.asarray(self.inbetween_locations[i]) == 1).sum() != num_ones:
                raise ValueError('The number of ones for each element in inbetween_locations must be the same.')

    def get_tf_record_names(self):
        return sorted(glob.glob(self._get_tf_record_pattern()))

    def init_data(self, session):
        session.run(self.iterator.initializer)

    def load(self, session, repeat=False, shuffle=False, initializable=False, max_num_elements=None):
        """
        :param session: tf Session.
        :param repeat: Whether to call repeat on the tf DataSet.
        :param shuffle: Whether to shuffle on the tf DataSet.
        :param initializable: Whether to use an initializable or a one_shot iterator.
                              If True, init_data must be called to use the DataSet.
        """
        with tf.name_scope(self.tf_record_name + '_dataset_ops'):
            for i in range(len(self.inbetween_locations)):
                inbetween_locations = self.inbetween_locations[i]
                dataset = self._load_for_inbetween_locations(inbetween_locations, shuffle)
                self.dataset = dataset if i == 0 else self.dataset.concatenate(dataset)

            if max_num_elements is not None:
                assert max_num_elements >= 0
                self.dataset = self.dataset.take(max_num_elements)

            buffer_size = 100
            if shuffle:
                self.dataset = self.dataset.shuffle(buffer_size=buffer_size)

            self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.prefetch(buffer_size=1)

            if repeat:
                self.dataset = self.dataset.repeat()

            if initializable:
                self.iterator = self.dataset.make_initializable_iterator()
            else:
                self.iterator = self.dataset.make_one_shot_iterator()

        self.handle = session.run(self.iterator.string_handle())

    def get_output_shapes(self):
        return self.dataset.output_shapes

    def get_output_types(self):
        return self.dataset.output_types

    def get_feed_dict_value(self):
        return self.handle

    def _get_tf_record_pattern(self):
        return os.path.join(self.directory, '*' + self.tf_record_name)

    def _load_for_inbetween_locations(self, inbetween_locations, shuffle):
        """
        :param inbetween_locations: An element of self.inbetween_locations.
        :param shuffle: Whether to shuffle the shards.
        :return: Tensorflow dataset object.
        """
        def _parse_function(example_proto):
            features = {
                SHOT_LEN: tf.FixedLenFeature((), tf.int64, default_value=0),
                SHOT: tf.VarLenFeature(tf.string),
                HEIGHT: tf.FixedLenFeature((), tf.int64, default_value=0),
                WIDTH: tf.FixedLenFeature((), tf.int64, default_value=0)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            shot_len = tf.reshape(tf.cast(parsed_features[SHOT_LEN], tf.int32), ())
            H = tf.reshape(tf.cast(parsed_features[HEIGHT], tf.int32), ())
            W = tf.reshape(tf.cast(parsed_features[WIDTH], tf.int32), ())

            shot_bytes = tf.sparse_tensor_to_dense(parsed_features[SHOT], default_value=tf.as_string(0))
            shot = tf.map_fn(lambda bytes: tf.image.decode_image(bytes), shot_bytes, dtype=(tf.uint8))
            shot = tf.image.convert_image_dtype(shot, tf.float32)
            shot = tf.reshape(shot, (shot_len, H, W, 3))

            # TODO: We should randomly augment triplets with timestamps like [0.0, 1.0, 1.0], or [0.0, 0.0, 1.0].
            # Decompose each shot into sequences of consecutive images.
            slice_locations = [1] + inbetween_locations + [1]
            return sliding_window_slice(shot, slice_locations)

        # Shuffle filenames.
        # Ideas taken from: https://github.com/tensorflow/tensorflow/issues/14857
        dataset = tf.data.Dataset.list_files(self._get_tf_record_pattern(), shuffle=shuffle)
        dataset = tf.data.TFRecordDataset(dataset)

        # Parse sequences.
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

        def _crop(sequence):
            s = tf.shape(sequence)
            sequence_len, H, W = s[0], s[1], s[2]
            crop_h, crop_w = self.crop_size
            if self.do_augment:
                sequence = tf.random_crop(sequence, [sequence_len, crop_h, crop_w, 3])
            else:
                crop_top = tf.cast(H / 2 - crop_h / 2, tf.int32)
                crop_left = tf.cast(W / 2 - crop_w / 2, tf.int32)
                sequence = tf.image.crop_to_bounding_box(sequence, crop_top, crop_left, crop_h, crop_w)
            return sequence

        def _flip(sequence):
            # Randomly flip in temporal, y, and x axes.
            sequence = tf.cond(tf_coin_flip(0.5)[0], lambda: tf.reverse(sequence, [0]), lambda: sequence)
            sequence = tf.cond(tf_coin_flip(0.5)[0], lambda: tf.reverse(sequence, [1]), lambda: sequence)
            sequence = tf.cond(tf_coin_flip(0.5)[0], lambda: tf.reverse(sequence, [2]), lambda: sequence)
            return sequence

        dataset = dataset.map(_crop, num_parallel_calls=multiprocessing.cpu_count())
        if self.do_augment:
            dataset = dataset.map(_flip, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.map(_add_timing, num_parallel_calls=multiprocessing.cpu_count())
        return dataset
