import os
import os.path
import unittest
from data.flow.flow_data import FlowDataSet
from data.flow.flow_data_preprocessor import FlyingChairsFlowDataPreprocessor
from data.flow.flow_data_test_base import TestFlowDataSet


class FlyingChairsTestPaths:
    full_directory = os.path.join('data', 'flow', 'test_data', 'flying_chairs', 'data')
    expected_image_a_paths = [os.path.join(full_directory, '06530_img1.ppm'),
                              os.path.join(full_directory, '06531_img1.ppm'),
                              os.path.join(full_directory, '06532_img1.ppm'),
                              os.path.join(full_directory, '06533_img1.ppm'),
                              os.path.join(full_directory, '22871_img1.ppm')]
    expected_image_b_paths = [os.path.join(full_directory, '06530_img2.ppm'),
                              os.path.join(full_directory, '06531_img2.ppm'),
                              os.path.join(full_directory, '06532_img2.ppm'),
                              os.path.join(full_directory, '06533_img2.ppm'),
                              os.path.join(full_directory, '22871_img2.ppm')]
    expected_flow_paths = [os.path.join(full_directory, '06530_flow.flo'),
                           os.path.join(full_directory, '06531_flow.flo'),
                           os.path.join(full_directory, '06532_flow.flo'),
                           os.path.join(full_directory, '06533_flow.flo'),
                           os.path.join(full_directory, '22871_flow.flo')]


class TestFlyingChairsFlowDataSet(TestFlowDataSet.TestCases):
    def setUp(self):
        super().setUp()

        data_directory = os.path.join('data', 'flow', 'test_data', 'flying_chairs')
        self.resolution = [384, 512]
        # No data augmentation so that the tests are deterministic.
        self.data_set = FlowDataSet(data_directory, batch_size=2, training_augmentations=False)
        self.data_set_preprocessor = FlyingChairsFlowDataPreprocessor(data_directory, validation_size=1, shard_size=2)

        # Test paths.
        self.expected_image_a_paths = FlyingChairsTestPaths.expected_image_a_paths
        self.expected_image_b_paths = FlyingChairsTestPaths.expected_image_b_paths
        self.expected_flow_paths = FlyingChairsTestPaths.expected_flow_paths


class TestFlyingChairsFlowDataSetWithCrop(TestFlowDataSet.TestCases):
    def setUp(self):
        super().setUp()

        data_directory = os.path.join('data', 'flow', 'test_data', 'flying_chairs')
        self.resolution = [384, 448]
        # No data augmentation so that the tests are deterministic.
        self.data_set = FlowDataSet(data_directory, batch_size=2,training_augmentations=False, crop_size=(384, 448))
        self.data_set_preprocessor = FlyingChairsFlowDataPreprocessor(data_directory, validation_size=1, shard_size=2)

        # Test paths.
        self.expected_image_a_paths = FlyingChairsTestPaths.expected_image_a_paths
        self.expected_image_b_paths = FlyingChairsTestPaths.expected_image_b_paths
        self.expected_flow_paths = FlyingChairsTestPaths.expected_flow_paths


if __name__ == '__main__':
    unittest.main()
