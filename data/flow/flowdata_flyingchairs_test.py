import os
import os.path
import unittest
from data.flow.flowdata import FlowDataSet
from data.flow.flowdata_test_base import TestFlowDataSet


class TestFlyingChairsFlowDataSet(TestFlowDataSet.TestCases):
    def setUp(self):
        super().setUp()

        data_directory = os.path.join('data', 'flow', 'test_data', 'flying_chairs')
        self.resolution = [384, 512]
        # No data augmentation so that the tests are deterministic.
        self.data_set = FlowDataSet(data_directory, batch_size=2, validation_size=1, training_augmentations=False,
                                    data_source=FlowDataSet.FLYING_CHAIRS)

        # Test paths.
        full_directory = os.path.join('data', 'flow', 'test_data', 'flying_chairs', 'data')
        self.expected_image_a_paths = [os.path.join(full_directory, '06530_img1.ppm'),
                                       os.path.join(full_directory, '06531_img1.ppm'),
                                       os.path.join(full_directory, '06532_img1.ppm'),
                                       os.path.join(full_directory, '06533_img1.ppm'),
                                       os.path.join(full_directory, '22871_img1.ppm')]
        self.expected_image_b_paths = [os.path.join(full_directory, '06530_img2.ppm'),
                                       os.path.join(full_directory, '06531_img2.ppm'),
                                       os.path.join(full_directory, '06532_img2.ppm'),
                                       os.path.join(full_directory, '06533_img2.ppm'),
                                       os.path.join(full_directory, '22871_img2.ppm')]
        self.expected_flow_paths = [os.path.join(full_directory, '06530_flow.flo'),
                                    os.path.join(full_directory, '06531_flow.flo'),
                                    os.path.join(full_directory, '06532_flow.flo'),
                                    os.path.join(full_directory, '06533_flow.flo'),
                                    os.path.join(full_directory, '22871_flow.flo')]


if __name__ == '__main__':
    unittest.main()
