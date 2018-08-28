import glob
import multiprocessing
import os.path
from common.utils.data import *
from common.utils.img import tf_random_crop, tf_image_augmentation
from common.utils.flow import tf_random_flip_flow, tf_random_scale_flow
from data.dataset import DataSet


class FlowDataSet(DataSet):
    HEIGHT = 'height'
    WIDTH = 'width'
    IMAGE_A_RAW = 'image_a_raw'
    IMAGE_B_RAW = 'image_b_raw'
    FLOW_RAW = 'flow_raw'

    TRAIN_FILENAME = 'flowdataset_train.tfrecords'
    VALID_FILENAME = 'flowdataset_valid.tfrecords'

    def __init__(self, directory, batch_size=1, crop_size=None, training_augmentations=True, augmentation_config=None):
        """
        :param directory: Str. Directory of the dataset file structure and tf records.
        :param batch_size: Int.
        :param crop_size: Tuple of (int (H), int (W)). Size to crop the training examples to before feeding to network.
                          If None, then no cropping will be performed.
        :param training_augmentations: Whether to do live augmentations while training.
        :param augmentation_config: Configurations for data augmentation. If None, the default will be used.
        """
        super().__init__(directory, batch_size, training_augmentations=training_augmentations)

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
                'do_scaling': True, 'flip_hor': True, 'flip_ver': True,
                'do_flipping': True
            }

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

    def load(self, session):
        """
        Overridden.
        """
        with tf.name_scope('dataset_ops'):
            self.train_dataset = self._load_dataset(self._get_train_file_name_pattern(), True,
                                                    do_augmentations=self.training_augmentations)
            self.valid_dataset = self._load_dataset(self._get_valid_file_name_pattern(), False,
                                                    do_augmentations=False)

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
        return os.path.join(self.directory, '*' + self.TRAIN_FILENAME)

    def _get_valid_file_name_pattern(self):
        """
        :return: Str.
        """
        return os.path.join(self.directory, '*' + self.VALID_FILENAME)

    def _load_dataset(self, filename_pattern, repeat, do_augmentations=False):
        """
        :param filename_pattern: Str. Pattern for globbing file names.
        :param repeat: Bool. Whether to repeat the dataset indefinitely.
        :param do_augmentations: Bool. Whether to do image augmentations.
        :return: Tensorflow dataset object.
        """
        def _parse_function(example_proto):
            features = {
                FlowDataSet.HEIGHT: tf.FixedLenFeature((), tf.int64, default_value=0),
                FlowDataSet.WIDTH: tf.FixedLenFeature((), tf.int64, default_value=0),
                FlowDataSet.IMAGE_A_RAW: tf.FixedLenFeature((), tf.string),
                FlowDataSet.IMAGE_B_RAW: tf.FixedLenFeature((), tf.string),
                FlowDataSet.FLOW_RAW: tf.FixedLenFeature((), tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            H = tf.reshape(tf.cast(parsed_features[FlowDataSet.HEIGHT], tf.int32), ())
            W = tf.reshape(tf.cast(parsed_features[FlowDataSet.WIDTH], tf.int32), ())
            image_a = tf.cast(tf.decode_raw(parsed_features[FlowDataSet.IMAGE_A_RAW], tf.uint8), tf.float32) / 255.0
            image_a = tf.reshape(image_a, [H, W, 3])
            image_b = tf.cast(tf.decode_raw(parsed_features[FlowDataSet.IMAGE_B_RAW], tf.uint8), tf.float32) / 255.0
            image_b = tf.reshape(image_b, [H, W, 3])
            flow = tf.decode_raw(parsed_features[FlowDataSet.FLOW_RAW], tf.float32)
            flow = tf.reshape(flow, [H, W, 2])

            # Cropping augmentation.
            image_a, image_b, flow = tf_random_crop([image_a, image_b, flow], self.crop_size)

            if do_augmentations:
                # Basic image augmentations.
                image_a, image_b = tf_image_augmentation([image_a, image_b], self.config)
                if self.config['do_scaling']:
                    # Flip randomly in unison.
                    flow, images = tf_random_flip_flow(flow, [image_a, image_b], flip_hor=self.config['flip_hor'],
                                                       flip_ver=self.config['flip_ver'])
                    image_a, image_b = images
                if self.config['do_flipping']:
                    # Scale randomly in unison.
                    flow, images = tf_random_scale_flow(flow, [image_a, image_b], self.config)
                    image_a, image_b = images

            return image_a, image_b, flow

        files = tf.data.Dataset.list_files(filename_pattern, shuffle=True)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=min(4, self.batch_size))
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(_parse_function, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)
        return dataset
