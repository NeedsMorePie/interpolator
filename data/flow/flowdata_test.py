import os
import os.path
import tensorflow as tf
import unittest
from data.flow.flowdata import FlowDataSet


class TestFlowDataSet(unittest.TestCase):
    def setUp(self):
        self.data_directory = os.path.join('data', 'flow', 'test_data')
        self.flow_directory = os.path.join(self.data_directory, 'test_flows')
        self.image_directory = os.path.join(self.data_directory, 'test_images')
        self.data_set = FlowDataSet(self.data_directory, batch_size=2, validation_size=1)

        # Test paths.
        self.expected_image_paths = [os.path.join(self.image_directory, 'set_a', 'image_0000.png'),
                                     os.path.join(self.image_directory, 'set_a', 'image_0001.png'),
                                     os.path.join(self.image_directory, 'set_a', 'image_0002.png'),
                                     os.path.join(self.image_directory, 'set_a', 'image_0003.png'),
                                     os.path.join(self.image_directory, 'set_b', 'image_0001.png')]
        self.expected_flow_paths = [os.path.join(self.flow_directory, 'set_a', 'flow_0000.flo'),
                                    os.path.join(self.flow_directory, 'set_a', 'flow_0001.flo'),
                                    os.path.join(self.flow_directory, 'set_a', 'flow_0002.flo'),
                                    os.path.join(self.flow_directory, 'set_a', 'flow_0003.flo'),
                                    os.path.join(self.flow_directory, 'set_b', 'flow_0001.flo')]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_data_reading(self):
        """
        Tests that we can properly read a dataset into numpy arrays.
        """
        image_paths, flow_paths = self.data_set._get_data_paths()

        self.assertListEqual(image_paths, self.expected_image_paths)
        self.assertListEqual(flow_paths, self.expected_flow_paths)

        # Test reading.
        images, flows = self.data_set._read_from_data_paths(image_paths, flow_paths)
        for image, flow in zip(images, flows):
            self.assertTupleEqual(image.shape, (436, 1024, 3))
            self.assertTupleEqual(flow.shape, (436, 1024, 2))

    def test_data_read_write(self):
        self.data_set.preprocess_raw(shard_size=2)
        output_paths = self.data_set.get_train_file_names() + self.data_set.get_validation_file_names()
        [self.assertTrue(os.path.isfile(output_path)) for output_path in output_paths]
        # The train set should have been sharded, so there should be 3 files.
        self.assertEquals(len(output_paths), 3)

        self.data_set.load()
        next_images, next_flows = self.data_set.get_next_train_batch()
        images, flows = self.sess.run([next_images, next_flows])
        self.assertTupleEqual(images.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))
        images, flows = self.sess.run([next_images, next_flows])
        self.assertTupleEqual(images.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))

        # Validation data size is 1, so even though the dataset batch size is 2, the validation batch size is 1.
        next_images, next_flows = self.data_set.get_next_validation_batch()
        images, flows = self.sess.run([next_images, next_flows])
        self.assertTupleEqual(images.shape, (1, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (1, 436, 1024, 2))

    def tearDown(self):
        output_paths = self.data_set.get_train_file_names() + self.data_set.get_validation_file_names()
        for output_path in output_paths:
            if os.path.isfile(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
