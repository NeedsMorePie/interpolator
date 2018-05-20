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
        self.data_set = FlowDataSet(self.data_directory)
        self.output_path = os.path.join(self.data_directory, self.data_set.get_processed_file_name())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_data_reading(self):
        """
        Tests that we can properly read a dataset into numpy arrays.
        """
        image_paths, flow_paths = self.data_set._get_data_paths()

        # Test paths.
        expected_image_paths = [os.path.join(self.image_directory, 'set_a', 'image_0000.png'),
                                os.path.join(self.image_directory, 'set_a', 'image_0001.png'),
                                os.path.join(self.image_directory, 'set_b', 'image_0001.png')]
        expected_flow_paths = [os.path.join(self.flow_directory, 'set_a', 'flow_0000.flo'),
                               os.path.join(self.flow_directory, 'set_a', 'flow_0001.flo'),
                               os.path.join(self.flow_directory, 'set_b', 'flow_0001.flo')]
        self.assertListEqual(image_paths, expected_image_paths)
        self.assertListEqual(flow_paths, expected_flow_paths)

        # Test reading.
        images, flows = self.data_set._read_from_data_paths(image_paths, flow_paths)
        for image, flow in zip(images, flows):
            self.assertTupleEqual(image.shape, (436, 1024, 3))
            self.assertTupleEqual(flow.shape, (436, 1024, 2))

    def test_data_read_write(self):
        data_set = FlowDataSet(self.data_directory, 2)

        self.assertFalse(os.path.isfile(self.output_path))
        data_set.preprocess_raw()
        self.assertTrue(os.path.isfile(self.output_path))

        data_set.load()
        next_images, next_flows = data_set.get_next_batch()
        images, flows = self.sess.run([next_images, next_flows])
        self.assertTupleEqual(images.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))

        # Second run should wrap and have a batch size of 1.
        images, flows = self.sess.run([next_images, next_flows])
        self.assertTupleEqual(images.shape, (1, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (1, 436, 1024, 2))

        # Third run should be batch size of 2 again.
        images, flows = self.sess.run([next_images, next_flows])
        self.assertTupleEqual(images.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))

    def tearDown(self):
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)


if __name__ == '__main__':
    unittest.main()
