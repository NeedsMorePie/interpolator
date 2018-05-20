import glob
import multiprocessing
import os.path
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
        # Tensorflow dataset objects.
        self.train_dataset = None
        self.validation_dataset = None
        # Tensors.
        self.next_train_images_a = None
        self.next_train_images_b = None
        self.next_train_flows = None
        self.next_validation_images_a = None
        self.next_validation_images_b = None
        self.next_validation_flows = None

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
        image_a_paths, image_b_paths, flow_paths = self._get_data_paths()
        self._convert_to_tf_record(image_a_paths, image_b_paths, flow_paths, shard_size)

    def load(self):
        """
        Overridden.
        """
        self.train_dataset = self._load_dataset(self.get_train_file_names())
        self.validation_dataset = self._load_dataset(self.get_validation_file_names())

        iterator = self.train_dataset.make_one_shot_iterator()
        self.next_train_images_a, self.next_train_images_b, self.next_train_flows = iterator.get_next()

        iterator = self.validation_dataset.make_one_shot_iterator()
        self.next_validation_images_a, self.next_validation_images_b, self.next_validation_flows = iterator.get_next()

    def get_next_train_batch(self):
        """
        Overridden.
        """
        return self.next_train_images_a, self.next_train_images_b, self.next_train_flows

    def get_next_validation_batch(self):
        """
        Overridden.
        """
        return self.next_validation_images_a, self.next_validation_images_b, self.next_validation_flows

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
        # Make sure the pairs are all under the same directory.
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

        def _write(filename, iter_range):
            sharded_iter_ranges = create_shard_ranges(iter_range, shard_size)

            Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading")(
                delayed(_write_shard)(shard_id, shard_range, image_a_paths, image_b_paths,
                                      flow_paths, filename, self.directory)
                for shard_id, shard_range in enumerate(sharded_iter_ranges)
            )

        valid_start_idx = len(image_a_paths) - self.validation_size
        _write(self.train_filename, range(0, valid_start_idx))
        _write(self.valid_filename, range(valid_start_idx, len(image_a_paths)))

    def _load_dataset(self, file_paths):
        """
        :param file_path: String. TfRecord file path.
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

        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset


def _write_shard(shard_id, shard_range, image_a_paths, image_b_paths, flow_paths, filename, directory):
    """
    :param shard_id: Index of the shard.
    :param shard_range: Iteration range of the shard.
    :param image_paths: Path of all images.
    :param flow_paths: Path of all flows.
    :param filename: Base name of the output shard.
    :param directory: Output directory.
    :return: Nothing.
    """
    images_a = [read_image(image_a_paths[i], as_float=True) for i in shard_range]
    images_b = [read_image(image_b_paths[i], as_float=True) for i in shard_range]
    flows = [read_flow_file(flow_paths[i]) for i in shard_range]

    writer = tf.python_io.TFRecordWriter(os.path.join(directory, str(shard_id) + '_' + filename))
    for image_a, image_b, flow in zip(images_a, images_b, flows):
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
