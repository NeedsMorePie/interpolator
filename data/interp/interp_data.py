import glob
import multiprocessing
import os.path
import random
import numpy as np
import json
from data.dataset import DataSet
from data.interp.interp_data_reader import InterpDataSetReader
from joblib import Parallel, delayed
from utils.data import *
from utils.img import read_image

SHOT_LEN = 'shot_len'
SHOT = 'shot'


class InterpDataSet(DataSet):
    def __init__(self, tf_record_directory, inbetween_locations, batch_size=1, maximum_shot_len=10):
        """
        :param inbetween_locations: A list of lists. Each element specifies where inbetweens will be placed,
                                    and each configuration will appear with uniform probability.
                                    For example, let a single element in the list be [0, 1, 0].
                                    With this, dataset elements will be sequences of 3 ordered frames,
                                    where the middle (inbetween) frame is 2 frames away from the first and last frames.
                                    The number of 1s must be the same for each list in this argument.
        :param maximum_shot_len: Video shots larger than this value will be broken up.
        """
        super().__init__(tf_record_directory, batch_size, validation_size=0)

        # Initialized during load().
        self.handle_placeholder = None  # Handle placeholder for switching between datasets.
        self.next_sequences = None  # Data iterator batch.
        self.next_sequence_timing = None  # Data iterator batch.

        self.tf_record_directory = tf_record_directory
        self.maximum_shot_len = maximum_shot_len
        self.inbetween_locations = inbetween_locations
        self.train_tf_record_name = 'interp_dataset_train.tfrecords'
        self.validation_tf_record_name = 'interp_dataset_validation.tfrecords'
        self.train_data = InterpDataSetReader(self.tf_record_directory, inbetween_locations,
                                              self.train_tf_record_name, batch_size=batch_size)
        self.validation_data = InterpDataSetReader(self.tf_record_directory, inbetween_locations,
                                                   self.validation_tf_record_name, batch_size=batch_size)
    def get_tf_record_dir(self):
        return self.tf_record_directory

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

    def preprocess_raw(self, raw_directory, shard_size, validation_size=0):
        """
        Processes the data in raw_directory to the tf_record_directory.
        :param raw_directory: The directory to the images to process.
        :param validation_size: The TfRecords will be partitioned such that, if possible,
                                this number of validation sequences can be used for validation.
        """
        if self.verbose:
            print('Checking directory for data.')
        image_paths = self._get_data_paths(raw_directory)
        self._convert_to_tf_record(image_paths, shard_size, validation_size)

        # Keep track of the validation size, as the amount of sequences that we write may allow slightly more.
        json_path = os.path.join(self.tf_record_directory, 'val_split.json')
        with open(json_path, 'w') as f:
            json_data = {'validation_size': validation_size}
            json.dump(json_data, f)

    def load(self, session):
        """
        Overriden.
        """
        with tf.name_scope('interp_data'):

            json_path = os.path.join(self.tf_record_directory, 'val_split.json')
            with open(json_path) as f:
                json_data = json.load(f)
                maximum_validation_size = json_data['validation_size']

            self.train_data.load(session, repeat=True, shuffle=True)
            self.validation_data.load(session, repeat=False, shuffle=False,
                                      initializable=True, max_num_elements=maximum_validation_size)

            self.handle_placeholder = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle_placeholder, self.train_data.get_output_types(), self.train_data.get_output_shapes())
            self.next_sequences, self.next_sequence_timing = self.iterator.get_next()

    def get_next_batch(self):
        return self.next_sequences, self.next_sequence_timing

    def get_train_feed_dict(self):
        """
        Overridden.
        """
        return {self.handle_placeholder: self.train_data.get_feed_dict_value()}

    def get_validation_feed_dict(self):
        """
        Overriden.
        """
        return {self.handle_placeholder: self.validation_data.get_feed_dict_value()}

    def init_validation_data(self, session):
        """
        Overridden
        """
        self.validation_data.init_data(session)

    def _get_data_paths(self, raw_directory):
        """
        :param raw_directory: The directory to the images to process.
        :return: List of list of image names, where image_paths[0][0] is the first image in the first video shot.
        """
        raise NotImplementedError

    def _process_image(self, filename):
        """
        Reads from and processes the file.
        :param filename: String. Full path to the image file.
        :return: The bytes that will be saved to the TFRecords.
                 Must be readable with tf.image.decode_image.
        """
        raise NotImplementedError

    def _convert_to_tf_record(self, image_paths, shard_size, validation_size):
        """
        :param image_paths: List of list of image names,
                            where image_paths[0][0] is the first image in the first video shot.
        :return: Nothing.
        """
        random.shuffle(image_paths)

        if not os.path.exists(self.tf_record_directory):
            os.mkdir(self.tf_record_directory)

        def _write(filename, iter_range, image_paths):
            if self.verbose:
                print('Writing', len(iter_range), 'data examples to the', filename, 'dataset.')

            sharded_iter_ranges = create_shard_ranges(iter_range, shard_size)

            Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading")(
                delayed(_write_shard)(shard_id, shard_range, image_paths, filename,
                                      self.tf_record_directory, self._process_image, self.verbose)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        image_paths = self._enforce_maximum_shot_len(image_paths)
        val_paths, train_paths = self._split_for_validation(image_paths, validation_size)
        image_paths = val_paths + train_paths
        train_start_idx = len(val_paths)
        _write(self.validation_tf_record_name, range(0, train_start_idx), image_paths)
        _write(self.train_tf_record_name, range(train_start_idx, len(image_paths)), image_paths)

    def _enforce_maximum_shot_len(self, image_paths):
        """
        :param image_paths: List of list of image names,
                            where image_paths[0][0] is the first image in the first video shot.
        :return: List in the same format as image_paths,
                 where len(return_value)[i] for all i <= self.maximum_shot_len.
        """
        cur_len = len(image_paths)
        i = 0
        while i < cur_len:
            if len(image_paths[i]) > self.maximum_shot_len:
                part_1 = image_paths[i][:self.maximum_shot_len]
                part_2 = image_paths[i][self.maximum_shot_len:]
                image_paths = image_paths[:i] + [part_1] + [part_2] + image_paths[i+1:]
                cur_len += 1
            i += 1
        return image_paths

    def _split_for_validation(self, image_paths, validation_size):
        """
        :param image_paths: List of list of image names,
                            where image_paths[0][0] is the first image in the first video shot.
        :param validation_size: The split will guarantee that at there will be at least this many validation elements.
        :return: (validation_image_paths, train_image_paths), where both have the same structure as image_paths.
        """
        if validation_size == 0:
            return [], image_paths

        # Count the number of sequences that exist for a certain shot length.
        max_len = 0
        for spec in self.inbetween_locations:
            max_len = max(2 + len(spec), max_len)

        a = np.zeros(max_len + 1)
        for spec in self.inbetween_locations:
            a[2 + len(spec)] += 1

        for i in range(1, len(a)):
            a[i] += a[i-1]

        # Find the split indices.
        cur_samples = 0
        split_indices = (len(image_paths)-1, len(image_paths[-1])-1)
        for i in range(len(image_paths)):
            for j in range(len(image_paths[i])):
                cur_samples += a[min(j + 1, len(a) - 1)]
                if cur_samples >= validation_size:
                    split_indices = (i, j)
                    break
            if cur_samples >= validation_size:
                break

        i, j = split_indices
        val_split = []
        val_split += image_paths[:i]
        if len(image_paths[i][:j+1]) > 0:
            val_split.append(image_paths[i][:j+1])

        train_split = []
        if len(image_paths[i][j+1:]) > 0:
            train_split.append(image_paths[i][j+1:])
        train_split += image_paths[i+1:]

        return val_split, train_split


def _write_shard(shard_id, shard_range, image_paths, filename, directory, processor_fn, verbose):
    """
    :param shard_id: Index of the shard.
    :param shard_range: Iteration range of the shard.
    :param image_paths: List of list of image names.
    :param filename: Base name of the output shard.
    :param directory: Output directory.
    :param processor_fn: Function to read and process from filename with before saving to TFRecords.
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
            shot_raw.append(processor_fn(image_path))

        H = reference_image.shape[0]
        W = reference_image.shape[1]

        # Write to tf record.
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    SHOT_LEN: tf_int64_feature(len(shot_raw)),
                    SHOT: tf_bytes_list_feature(shot_raw),
                }))
        writer.write(example.SerializeToString())
    writer.close()
