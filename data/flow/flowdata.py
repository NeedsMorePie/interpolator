import glob
import multiprocessing
import numpy as np
import os.path
import random
from data.dataset import DataSet
from joblib import Parallel, delayed
from utils.data import *
from utils.flow import read_flow_file, tf_random_flip_flow, tf_random_scale_flow
from utils.img import read_image, tf_random_crop, tf_image_augmentation


HEIGHT = 'height'
WIDTH = 'width'
IMAGE_A_RAW = 'image_a_raw'
IMAGE_B_RAW = 'image_b_raw'
FLOW_RAW = 'flow_raw'


class FlowDataSet(DataSet):
    # Data sources.
    SINTEL = 0
    FLYING_CHAIRS = 1
    FLYING_THINGS = 2

    def __init__(self, directory, batch_size=1, validation_size=1, crop_size=None, training_augmentations=True,
                 data_source=SINTEL, augmentation_config=None, max_flow=1000.0):
        """
        :param directory: Str. Directory of the dataset file structure and tf records.
        :param batch_size: Int.
        :param validation_size: Int. Number of examples to reserve for validation
        :param crop_size: Tuple of (int (H), int (W)). Size to crop the training examples to before feeding to network.
                          If None, then no cropping will be performed.
        :param training_augmentations: Whether to do live augmentations while training.
        :param data_source: Source of the data.
        :param augmentation_config: Configurations for data augmentation. If None, the default will be used.
        :param max_flow: Float. Maximum flow magnitude of the flow image. Any examples with flow magnitude greater than
            this will be ignored.
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
        self.next_images_a = None  # Data iterator batch.
        self.next_images_b = None  # Data iterator batch.
        self.next_flows = None  # Data iterator batch.

        self.crop_size = crop_size
        self.training_augmentations = training_augmentations
        self.data_source = data_source

        self.train_filename = 'flowdataset_train.tfrecords'
        self.valid_filename = 'flowdataset_valid.tfrecords'

        self.config = augmentation_config
        if augmentation_config is None:
            self.config = {
                'contrast_min': 0.8, 'contrast_max': 1.25,
                'gamma_min': 0.8, 'gamma_max': 1.25,
                'gain_min': 0.8, 'gain_max': 1.25,
                'brightness_stddev': 0.2,
                'hue_min': -0.3, 'hue_max': 0.3,
                'noise_stddev': 0.04,
                'scale_min': 0.5, 'scale_max': 2.0,
                'do_scaling': True,
                'do_flipping': True
            }
        self.max_flow = max_flow

    def get_train_file_names(self):
        """
        Overridden.
        """
        return glob.glob(self._get_train_file_name_pattern())

    def get_validation_file_names(self):
        """
        :return: List of string.
        """
        return glob.glob(self._get_valid_file_name_pattern())

    def preprocess_raw(self, shard_size):
        """
        Overridden.
        """
        if self.verbose:
            print('Checking directory for data...')
        image_a_paths, image_b_paths, flow_paths = self._get_data_paths()
        if self.verbose:
            print('Converting to tf records...')
        self._convert_to_tf_record(image_a_paths, image_b_paths, flow_paths, shard_size)

    def load(self, session, shuffle=False):
        """
        :param session: Tensorflow session.
        :param shuffle: Bool. Whether to shuffle individual data points. Note that the tf record files will be shuffled
            anyway.
        :return: Nothing.
        """
        with tf.name_scope('dataset_ops'):
            self.train_dataset = self._load_dataset(self._get_train_file_name_pattern(), True,
                                                    do_augmentations=self.training_augmentations, shuffle=shuffle)
            self.valid_dataset = self._load_dataset(self._get_valid_file_name_pattern(), False,
                                                    do_augmentations=False, shuffle=shuffle)

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

    def _get_train_file_name_pattern(self):
        """
        :return: Str.
        """
        return os.path.join(self.directory, '*' + self.train_filename)

    def _get_valid_file_name_pattern(self):
        """
        :return: Str.
        """
        return os.path.join(self.directory, '*' + self.valid_filename)

    def _get_data_paths(self):
        """
        :return: List of image_path strings, list of flow_path strings.
        """
        if self.data_source == self.SINTEL:
            return self._get_data_paths_sintel()
        elif self.data_source == self.FLYING_CHAIRS:
            return self._get_data_paths_flying_chairs()
        elif self.data_source == self.FLYING_THINGS:
            return self._get_data_paths_flying_things()
        return None

    def _get_data_paths_sintel(self):
        """
        Gets the paths of [image_a, image_b, flow] tuples from a typical Sintel flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        # Get sorted lists.
        images = glob.glob(os.path.join(self.directory, '**', '*.png'), recursive=True)
        flows = glob.glob(os.path.join(self.directory, '**', '*.flo'), recursive=True)
        if self.verbose:
            print('Sorting file paths...')
        images.sort()
        flows.sort()
        if self.verbose:
            print('Filtering file paths...')
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

    def _get_data_paths_flying_chairs(self):
        """
        Gets the paths of [image_a, image_b, flow] tuples from a typical flying chairs flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        images_a = glob.glob(os.path.join(self.directory, '**', '*_img1.ppm'), recursive=True)
        if self.verbose:
            print('Sorting file paths...')
        images_a.sort()
        images_b = [image_a.replace('img1', 'img2') for image_a in images_a]
        flows = [image_a.replace('img1', 'flow').replace('ppm', 'flo') for image_a in images_a]
        return images_a, images_b, flows

    def _get_data_paths_flying_things(self):
        """
        Gets the paths of [image_a, image_b, flow] tuples from a typical flying things flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        # Get sorted lists.
        images = glob.glob(os.path.join(self.directory, '**', 'TRAIN', '**', 'left', '*.png'), recursive=True)
        flows = glob.glob(os.path.join(self.directory, '**', 'TRAIN', '**', 'into_future', 'left', '*.pfm'),
                          recursive=True)
        if self.verbose:
            print('Sorting file paths...')
        images.sort()
        flows.sort()
        assert len(images) == len(flows)
        if self.verbose:
            print('Filtering file paths...')
        # Make sure the tuples are all under the same directory.
        filtered_images_a = []
        filtered_images_b = []
        filtered_flows = []
        for i in range(len(images) - 1):
            directory_a = os.path.dirname(images[i])
            directory_b = os.path.dirname(images[i + 1])
            if directory_a == directory_b:
                filtered_images_a.append(images[i])
                filtered_images_b.append(images[i + 1])
                filtered_flows.append(flows[i])
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
                                      flow_paths, filename, self.directory, self.verbose, self.max_flow)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        valid_start_idx = len(image_a_paths) - self.validation_size
        _write(self.train_filename, range(0, valid_start_idx))
        _write(self.valid_filename, range(valid_start_idx, len(image_a_paths)))

    def _load_dataset(self, filename_pattern, repeat, do_augmentations=False, shuffle=False):
        """
        :param filename_pattern: Str. Pattern for globbing file names.
        :param repeat: Whether to repeat the dataset indefinitely.
        :param do_augmentations: Bool. Whether to do image augmentations.
        :param shuffle: Bool. Whether to shuffle the individual data points.
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
            image_a = tf.cast(tf.decode_raw(parsed_features[IMAGE_A_RAW], tf.uint8), tf.float32) / 255.0
            image_a = tf.reshape(image_a, [H, W, 3])
            image_b = tf.cast(tf.decode_raw(parsed_features[IMAGE_B_RAW], tf.uint8), tf.float32) / 255.0
            image_b = tf.reshape(image_b, [H, W, 3])
            flow = tf.decode_raw(parsed_features[FLOW_RAW], tf.float32)
            flow = tf.reshape(flow, [H, W, 2])

            # Cropping augmentation.
            image_a, image_b, flow = tf_random_crop([image_a, image_b, flow], self.crop_size)

            if do_augmentations:
                # Basic image augmentations.
                image_a, image_b = tf_image_augmentation([image_a, image_b], self.config)
                if self.config['do_scaling']:
                    # Flip randomly in unison.
                    flow, images = tf_random_flip_flow(flow, [image_a, image_b])
                    image_a, image_b = images
                if self.config['do_flipping']:
                    # Scale randomly in unison.
                    flow, images = tf_random_scale_flow(flow, [image_a, image_b], self.config)
                    image_a, image_b = images

            return image_a, image_b, flow

        dataset = tf.data.Dataset.list_files(filename_pattern, shuffle=True)
        dataset = tf.data.TFRecordDataset(dataset, compression_type='GZIP')
        if shuffle:
            if repeat:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.batch_size * 10))
            else:
                dataset = dataset.shuffle(buffer_size=self.batch_size * 10)
        elif repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(_parse_function, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)
        return dataset


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
                    HEIGHT: tf_int64_feature(H),
                    WIDTH: tf_int64_feature(W),
                    IMAGE_A_RAW: tf_bytes_feature(image_a_raw),
                    IMAGE_B_RAW: tf_bytes_feature(image_b_raw),
                    FLOW_RAW: tf_bytes_feature(flow_raw)
                }))
        writer.write(example.SerializeToString())
        num_examples_written += 1
    writer.close()

    if num_examples_written == 0:
        # Delete the file if nothing was written to it.
        if verbose:
            print(record_name, 'is empty')
        silently_remove_file(record_name)
