import glob
import os.path
import tensorflow as tf
from data.dataset import DataSet
from utils.flow import read_flow_file
from utils.img import read_image


HEIGHT = 'height'
WIDTH = 'width'
IMAGE_RAW = 'image_raw'
FLOW_RAW = 'flow_raw'


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

    def get_processed_file_names(self):
        """
        Overridden.
        """
        return ['flowdataset_train.tfrecords', 'flowdataset_valid.tfrecords']

    def preprocess_raw(self):
        """
        Overridden.
        """
        image_paths, flow_paths = self._get_data_paths()
        images, flows = self._read_from_data_paths(image_paths, flow_paths)
        self._convert_to_tf_record(images, flows)

    def load(self):
        """
        Overridden.
        """
        self.train_dataset = self._load_dataset([os.path.join(self.directory, self.get_processed_file_names()[0])])
        self.validation_dataset = self._load_dataset([os.path.join(self.directory, self.get_processed_file_names()[1])])

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

    def _read_from_data_paths(self, image_paths, flow_paths):
        """
        Reads images as np arrays between 0.0 and 1.0.
        Flows are not normalized.
        :param image_paths: List of image_path strings.
        :param flow_paths: List of flow_path strings.
        :return: List of image_np_arrays, list of flow_np_arrays.
        """
        return [read_image(image, as_float=True) for image in image_paths],\
               [read_flow_file(flow) for flow in flow_paths]

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _convert_to_tf_record(self, images, flows):
        """
        :param images: List of image_np_arrays.
        :param flows: List of flow_np_arrays.
        :return: Nothing.
        """
        assert len(images) > 0
        H = images[0].shape[0]
        W = images[0].shape[1]
        train_filename = os.path.join(self.directory, self.get_processed_file_names()[0])
        valid_filename = os.path.join(self.directory, self.get_processed_file_names()[1])

        def _write(filename, iter_range):
            with tf.python_io.TFRecordWriter(filename) as writer:
                for i in iter_range:
                    image_raw = images[i].tostring()
                    flow_raw = flows[i].tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                HEIGHT: self._int64_feature(H),
                                WIDTH: self._int64_feature(W),
                                IMAGE_RAW: self._bytes_feature(image_raw),
                                FLOW_RAW: self._bytes_feature(flow_raw)
                            }))
                    writer.write(example.SerializeToString())

        valid_start_idx = len(images) - self.validation_size
        _write(train_filename, range(0, valid_start_idx))
        _write(valid_filename, range(valid_start_idx, len(images)))

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
