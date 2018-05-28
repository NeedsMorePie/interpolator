import glob
import multiprocessing
import os.path
import random
from data.dataset import DataSet
from joblib import Parallel, delayed
from utils.data import *
from utils.flow import read_flow_file
from utils.img import read_image


HEIGHT = 'height'
WIDTH = 'width'
IMAGE_A_RAW = 'image_a_raw'
IMAGE_B_RAW = 'image_b_raw'
FLOW_RAW = 'flow_raw'


class FlowDataSet(DataSet):
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
        image_a_paths, image_b_paths, flow_paths = self._get_data_paths()
        self._convert_to_tf_record(image_a_paths, image_b_paths, flow_paths, shard_size)

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
        Gets the paths of [image_a, image_b, flow] tuples from a typical flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        # Get sorted lists.
        images = glob.glob(os.path.join(self.directory, '**', '*.png'), recursive=True)
        images.sort()
        flows = glob.glob(os.path.join(self.directory, '**', '*.flo'), recursive=True)
        flows.sort()
        # Make sure the tuples are all under the same directory.
        filtered_images_a = []
        filtered_images_b = []
        filtered_flows = []
        flow_idx = 0
        for i in range(len(images) - 1):
            directory_a = os.path.dirname(images[i])
            directory_b = os.path.dirname(images[i + 1])
            if directory_a == directory_b:
                filtered_images_a.append(images[i])
                filtered_images_b.append(images[i + 1])
                filtered_flows.append(flows[flow_idx])
                flow_idx += 1
        assert flow_idx == len(flows)
        return filtered_images_a, filtered_images_b, filtered_flows

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
                                      flow_paths, filename, self.directory, self.verbose)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        valid_start_idx = len(image_a_paths) - self.validation_size
        _write(self.train_filename, range(0, valid_start_idx))
        _write(self.valid_filename, range(valid_start_idx, len(image_a_paths)))

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
        if repeat:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(filenames)))
        else:
            dataset = dataset.shuffle(buffer_size=len(filenames))
        dataset = dataset.map(_parse_function, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size)
        return dataset


def _write_shard(shard_id, shard_range, image_a_paths, image_b_paths, flow_paths, filename, directory, verbose):
    """
    :param shard_id: Index of the shard.
    :param shard_range: Iteration range of the shard.
    :param image_paths: Path of all images.
    :param flow_paths: Path of all flows.
    :param filename: Base name of the output shard.
    :param directory: Output directory.
    :return: Nothing.
    """
    if verbose and len(shard_range) > 0:
        print('Writing to shard', shard_id, 'data points', shard_range[0], 'to', shard_range[-1])

    writer = tf.python_io.TFRecordWriter(os.path.join(directory, str(shard_id) + '_' + filename))
    for i in shard_range:
        # Read from file.
        image_a = read_image(image_a_paths[i], as_float=True)
        image_b = read_image(image_b_paths[i], as_float=True)
        flow = read_flow_file(flow_paths[i])

        # Write to tf record.
        H = image_a.shape[0]
        W = image_a.shape[1]
        image_a_raw = image_a.tostring()
        image_b_raw = image_b.tostring()
        flow_raw = flow.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    HEIGHT: tf_int64_feature(H),
                    WIDTH: tf_int64_feature(W),
                    IMAGE_A_RAW: tf_bytes_feature(image_a_raw),
                    IMAGE_B_RAW: tf_bytes_feature(image_b_raw),
                    FLOW_RAW: tf_bytes_feature(flow_raw)
                }))
        writer.write(example.SerializeToString())
    writer.close()
