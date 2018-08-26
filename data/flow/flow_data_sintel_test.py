import os
import os.path
import unittest
from data.flow.flow_data import FlowDataSet
from data.flow.flow_data_preprocessor import SintelFlowDataPreprocessor
from data.flow.flow_data_test_base import TestFlowDataSet


class TestSintelFlowDataSet(TestFlowDataSet.TestCases):
    def setUp(self):
        super().setUp()

        data_directory = os.path.join('data', 'flow', 'test_data', 'sintel')
        flow_directory = os.path.join(data_directory, 'test_flows')
        image_directory = os.path.join(data_directory, 'test_images')
        self.resolution = [436, 1024]
        # No data augmentation so that the tests are deterministic.
        self.data_set = FlowDataSet(data_directory, batch_size=2, training_augmentations=False)
        self.data_set_preprocessor = SintelFlowDataPreprocessor(data_directory, validation_size=1, shard_size=2)

        # Test paths.
        self.expected_image_a_paths = [os.path.join(image_directory, 'set_a', 'image_0000.png'),
                                       os.path.join(image_directory, 'set_a', 'image_0001.png'),
                                       os.path.join(image_directory, 'set_a', 'image_0002.png'),
                                       os.path.join(image_directory, 'set_a', 'image_0003.png'),
                                       os.path.join(image_directory, 'set_b', 'image_0001.png')]
        self.expected_image_b_paths = [os.path.join(image_directory, 'set_a', 'image_0001.png'),
                                       os.path.join(image_directory, 'set_a', 'image_0002.png'),
                                       os.path.join(image_directory, 'set_a', 'image_0003.png'),
                                       os.path.join(image_directory, 'set_a', 'image_0004.png'),
                                       os.path.join(image_directory, 'set_b', 'image_0002.png')]
        self.expected_flow_paths = [os.path.join(flow_directory, 'set_a', 'flow_0000.flo'),
                                    os.path.join(flow_directory, 'set_a', 'flow_0001.flo'),
                                    os.path.join(flow_directory, 'set_a', 'flow_0002.flo'),
                                    os.path.join(flow_directory, 'set_a', 'flow_0003.flo'),
                                    os.path.join(flow_directory, 'set_b', 'flow_0001.flo')]


if __name__ == '__main__':
    unittest.main()
