import unittest
import numpy as np
import tensorflow as tf
from utils.flow import read_flow_file, show_flow_image, get_flow_visualization, get_tf_flow_visualization
from utils.img import show_image


SHOW_SINTEL_TEST_IMAGES = False


class TestSintelFlowReader(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_flow_vis(self):
        flow_image = read_flow_file('utils/test_data/frame_0001.flo')
        self.assertTrue(flow_image is not None)

        expected_shape = np.asarray([436, 1024, 2], dtype=np.int)
        self.assertTrue(np.allclose(expected_shape, np.asarray(flow_image.shape)))
        if SHOW_SINTEL_TEST_IMAGES:
            show_flow_image(flow_image)

        flow_image = read_flow_file('utils/test_data/frame_0011.flo')
        self.assertTrue(flow_image is not None)

        self.assertTrue(np.allclose(expected_shape, np.asarray(flow_image.shape)))
        if SHOW_SINTEL_TEST_IMAGES:
            show_flow_image(flow_image)

    def test_tensorflow_vis(self):
        """
        Tests the tensorflow implementation and the batching against the numpy version.
        """
        flow_image = read_flow_file('utils/test_data/frame_0001.flo')
        flow_image_2 = read_flow_file('utils/test_data/frame_0011.flo')
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

        if SHOW_SINTEL_TEST_IMAGES:
            show_image(visualization[0])
            show_image(visualization[1])


if __name__ == '__main__':
    unittest.main()
