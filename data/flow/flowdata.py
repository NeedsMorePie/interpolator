import glob
import multiprocessing
import os.path
import tensorflow as tf
from data.dataset import DataSet
from joblib import Parallel, delayed
from utils.flow import read_flow_file
from utils.img import read_image


HEIGHT = 'height'
WIDTH = 'width'
IMAGE_RAW = 'image_raw'
FLOW_RAW = 'flow_raw'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_shard(shard_id, shard_range, image_paths, flow_paths, filename, directory):
    images = [read_image(image_paths[i], as_float=True) for i in shard_range]
    flows = [read_flow_file(flow_paths[i]) for i in shard_range]

    writer = tf.python_io.TFRecordWriter(os.path.join(directory, str(shard_id) + '_' + filename))
    for image, flow in zip(images, flows):
        H = image.shape[0]
        W = image.shape[1]
        image_raw = image.tostring()
        flow_raw = flow.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    HEIGHT: _int64_feature(H),
                    WIDTH: _int64_feature(W),
                    IMAGE_RAW: _bytes_feature(image_raw),
                    FLOW_RAW: _bytes_feature(flow_raw)
                }))
        writer.write(example.SerializeToString())
    writer.close()


class FlowDataSet(DataSet):
    def __init__(self, directory, batch_size=1, validation_size=1):
        super().__init__(directory, batch_size, validation_size)
        # Tensorflow dataset objects.
        self.train_dataset = None
        self.validation_dataset = None
        # Tensors.
        self.next_train_images = None
        self.next_train_flows = None
        self.next_validation_images = None
        self.next_validation_flows = None

    def get_train_file_names(self):
        """
        Overridden.
        """
        return glob.glob(os.path.join(self.directory, '*flowdataset_train.tfrecords'))

    def get_validation_file_names(self):
        """
        :return: List of string.
        """
        return glob.glob(os.path.join(self.directory, '*flowdataset_valid.tfrecords'))

    def preprocess_raw(self, shard_size):
        """
        Overridden.
        """
        image_paths, flow_paths = self._get_data_paths()
        self._convert_to_tf_record(image_paths, flow_paths, shard_size)

    def load(self):
        """
        Overridden.
        """
        self.train_dataset = self._load_dataset(self.get_train_file_names())
        self.validation_dataset = self._load_dataset(self.get_validation_file_names())

        iterator = self.train_dataset.make_one_shot_iterator()
        self.next_train_images, self.next_train_flows = iterator.get_next()

        iterator = self.validation_dataset.make_one_shot_iterator()
        self.next_validation_images, self.next_validation_flows = iterator.get_next()

    def get_next_train_batch(self):
        """
        Overridden.
        """
        return self.next_train_images, self.next_train_flows

    def get_next_validation_batch(self):
        """
        Overridden.
        """
        return self.next_validation_images, self.next_validation_flows

    def _get_data_paths(self):
        """
        Gets the paths of [image, flow] pairs from a typical flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        # Get sorted lists.
        images = glob.glob(os.path.join(self.directory, '**', '*.png'), recursive=True)
        images.sort()
        flows = glob.glob(os.path.join(self.directory, '**', '*.flo'), recursive=True)
        flows.sort()
        return images, flows

    def _create_shard_ranges(self, iter_range, shard_size):
        sharded_iter_ranges = []
        current_shard_ids = []
        first_example = True
        for i in iter_range:
            if i % shard_size == 0 and not first_example:
                sharded_iter_ranges.append(current_shard_ids)
                current_shard_ids = []
            first_example = False
            current_shard_ids.append(i)
        if len(current_shard_ids) != 0:
            sharded_iter_ranges.append(current_shard_ids)
        return sharded_iter_ranges

    def _convert_to_tf_record(self, image_paths, flow_paths, shard_size):
        """
        :param image_paths: List of image_path strings.
        :param flow_paths: List of flow_np_path strings.
        :param shard_size: Maximum number of examples in each shard.
        :return: Nothing.
        """
        assert len(image_paths) == len(flow_paths)
        train_filename = 'flowdataset_train.tfrecords'
        valid_filename = 'flowdataset_valid.tfrecords'

        def _write(filename, iter_range):
            sharded_iter_ranges = self._create_shard_ranges(iter_range, shard_size)

            Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading")(
                delayed(_write_shard)(shard_id, shard_range, image_paths, flow_paths, filename, self.directory)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        valid_start_idx = len(image_paths) - self.validation_size
        _write(train_filename, range(0, valid_start_idx))
        _write(valid_filename, range(valid_start_idx, len(image_paths)))

    def _load_dataset(self, file_paths):
        """
        :param file_path: String. TfRecord file path.
        :return: Tensorflow dataset object.
        """
        def _parse_function(example_proto):
            features = {
                HEIGHT: tf.FixedLenFeature((), tf.int64, default_value=0),
                WIDTH: tf.FixedLenFeature((), tf.int64, default_value=0),
                IMAGE_RAW: tf.FixedLenFeature((), tf.string),
                FLOW_RAW: tf.FixedLenFeature((), tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            H = tf.reshape(tf.cast(parsed_features[HEIGHT], tf.int32), ())
            W = tf.reshape(tf.cast(parsed_features[WIDTH], tf.int32), ())
            image = tf.decode_raw(parsed_features[IMAGE_RAW], tf.float32)
            image = tf.reshape(image, [H, W, 3])
            flow = tf.decode_raw(parsed_features[FLOW_RAW], tf.float32)
            flow = tf.reshape(flow, [H, W, 2])
            return image, flow

        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset
