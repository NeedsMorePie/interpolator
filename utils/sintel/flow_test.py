import numpy as np
import unittest
from utils.sintel.flow import read_flow_file
from utils.img_utils import show_flow_image


SHOW_SINTEL_TEST_IMAGES = False


class TestSintelFlowReader(unittest.TestCase):
    def runTest(self):
        flow_image = read_flow_file('utils/sintel/test_data/frame_0001.flo')
        self.assertTrue(flow_image is not None)

        expected_shape = np.asarray([436, 1024, 2], dtype=np.int)
        self.assertTrue(np.allclose(expected_shape, np.asarray(flow_image.shape)))
        if SHOW_SINTEL_TEST_IMAGES:
            show_flow_image(flow_image)

        flow_image = read_flow_file('utils/sintel/test_data/frame_0011.flo')
        self.assertTrue(flow_image is not None)

        self.assertTrue(np.allclose(expected_shape, np.asarray(flow_image.shape)))
        if SHOW_SINTEL_TEST_IMAGES:
            show_flow_image(flow_image)


if __name__ == '__main__':
    unittest.main()
