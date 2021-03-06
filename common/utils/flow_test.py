import unittest
import numpy as np
import tensorflow as tf
from common.utils.data import silently_remove_file
from common.utils.img import show_image
from common.utils.flow import read_flow_file, write_flow_file, show_flow_image, \
    get_flow_visualization, get_tf_flow_visualization


SHOW_FLOW_TEST_IMAGES = False


class TestSintelFlowReader(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_flow_vis(self):
        flow_image = read_flow_file('common/utils/test_data/frame_0001.flo')
        self.assertTrue(flow_image is not None)

        expected_shape = (436, 1024, 2)
        self.assertTupleEqual((436, 1024, 2), flow_image.shape)
        if SHOW_FLOW_TEST_IMAGES:
            show_flow_image(flow_image)

        flow_image = read_flow_file('common/utils/test_data/frame_0011.flo')
        self.assertTrue(flow_image is not None)

        self.assertTupleEqual(expected_shape, flow_image.shape)
        if SHOW_FLOW_TEST_IMAGES:
            show_flow_image(flow_image)

    def test_tensorflow_vis(self):
        """
        Tests the tensorflow implementation and the batching against the numpy version.
        """
        flow_image = read_flow_file('common/utils/test_data/frame_0001.flo')
        flow_image_2 = read_flow_file('common/utils/test_data/frame_0011.flo')
        flow_ph = tf.placeholder(shape=[None, flow_image.shape[0], flow_image.shape[1], flow_image.shape[2]],
                                 dtype=tf.float32)
        visualization_tensor = get_tf_flow_visualization(flow_ph)
        feed_dict = {
            flow_ph: np.stack([flow_image, flow_image_2], axis=0)
        }
        visualization = self.sess.run(visualization_tensor, feed_dict=feed_dict)

        target_visualization = get_flow_visualization(flow_image) / 255.0
        target_visualization_2 = get_flow_visualization(flow_image_2) / 255.0
        self.assertTrue(np.allclose(target_visualization, visualization[0], atol=0.06))
        self.assertTrue(np.allclose(target_visualization_2, visualization[1], atol=0.06))

        if SHOW_FLOW_TEST_IMAGES:
            show_image(visualization[0])
            show_image(visualization[1])


class TestPFMFlowReader(unittest.TestCase):
    def test_pfm_read(self):
        flow_image = read_flow_file('common/utils/test_data/flow.pfm')
        self.assertTrue(flow_image is not None)
        self.assertTupleEqual((540, 960, 2), flow_image.shape)
        if SHOW_FLOW_TEST_IMAGES:
            show_flow_image(flow_image)


class TestFlowWriter(unittest.TestCase):
    def setUp(self):
        self.out_file = 'common/utils/test_data/temp_flow_write_test_file.flo'

    def test_write(self):
        flow_image_expected = read_flow_file('common/utils/test_data/frame_0001.flo')
        write_flow_file(self.out_file, flow_image_expected)
        flow_image = read_flow_file(self.out_file)
        self.assertTrue(np.allclose(flow_image_expected, flow_image))
        if SHOW_FLOW_TEST_IMAGES:
            show_flow_image(flow_image)

    def tearDown(self):
        silently_remove_file(self.out_file)


if __name__ == '__main__':
    unittest.main()
