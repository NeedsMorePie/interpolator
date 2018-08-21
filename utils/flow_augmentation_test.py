import unittest

import cv2
import numpy as np

from pwcnet.warp.warp import *
from utils.flow import read_flow_file, tf_scale_flow, tf_flip_flow
from utils.img import read_image, show_image

SHOW_AUGMENTATION_DEBUG_IMAGES = False


class TestFlowAugmentation(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Load from files.
        s = 0.5  # Scale.
        self.flow_ab = cv2.resize(read_flow_file('utils/test_data/flow_ab.flo'), (0, 0), fx=s, fy=s) * s
        self.img_a = cv2.resize(read_image('utils/test_data/image_a.png', as_float=True), (0, 0), fx=s, fy=s)
        self.img_b = cv2.resize(read_image('utils/test_data/image_b.png', as_float=True), (0, 0), fx=s, fy=s)
        self.flow_cd = cv2.resize(read_flow_file('utils/test_data/flow_cd.flo'), (0, 0), fx=s, fy=s) * s
        self.img_c = cv2.resize(read_image('utils/test_data/image_c.png', as_float=True), (0, 0), fx=s, fy=s)
        self.img_d = cv2.resize(read_image('utils/test_data/image_d.png', as_float=True), (0, 0), fx=s, fy=s)
        self.tf_true = tf.constant(True)
        self.tf_false = tf.constant(False)

    def test_scale_vertical(self):
        flow_ab, img_a, img_b = self.scale_immediate(self.flow_ab, self.img_a, self.img_b, 1.4)
        flow_cd, img_c, img_d = self.scale_immediate(self.flow_cd, self.img_c, self.img_d, 1.4)
        self.run_case(flow_ab, img_a, img_b, flow_cd, img_c, img_d)

    def test_scale_horizontal(self):
        flow_ab, img_a, img_b = self.scale_immediate(self.flow_ab, self.img_a, self.img_b, 0.6)
        flow_cd, img_c, img_d = self.scale_immediate(self.flow_cd, self.img_c, self.img_d, 0.6)
        self.run_case(flow_ab, img_a, img_b, flow_cd, img_c, img_d)

    def test_flip_horizontal(self):
        flow_ab, img_a, img_b = self.flip_immediate(self.flow_ab, self.img_a, self.img_b, self.tf_true, self.tf_false)
        flow_cd, img_c, img_d = self.flip_immediate(self.flow_cd, self.img_c, self.img_d, self.tf_true, self.tf_false)
        self.run_case(flow_ab, img_a, img_b, flow_cd, img_c, img_d)

    def test_flip_vertical(self):
        flow_ab, img_a, img_b = self.flip_immediate(self.flow_ab, self.img_a, self.img_b, self.tf_false, self.tf_true)
        flow_cd, img_c, img_d = self.flip_immediate(self.flow_cd, self.img_c, self.img_d, self.tf_false, self.tf_true)
        self.run_case(flow_ab, img_a, img_b, flow_cd, img_c, img_d)

    def test_flip_both(self):
        flow_ab, img_a, img_b = self.flip_immediate(self.flow_ab, self.img_a, self.img_b, self.tf_true, self.tf_true)
        flow_cd, img_c, img_d = self.flip_immediate(self.flow_cd, self.img_c, self.img_d, self.tf_true, self.tf_true)
        self.run_case(flow_ab, img_a, img_b, flow_cd, img_c, img_d)

    def test_scale_and_flip(self):
        flow_ab, img_a, img_b = self.flip_immediate(self.flow_ab, self.img_a, self.img_b, self.tf_true, self.tf_true)
        flow_cd, img_c, img_d = self.flip_immediate(self.flow_cd, self.img_c, self.img_d, self.tf_true, self.tf_true)
        flow_ab, img_a, img_b = self.scale_immediate(flow_ab, img_a, img_b, 1.4)
        flow_cd, img_c, img_d = self.scale_immediate(flow_cd, img_c, img_d, 1.4)
        self.run_case(flow_ab, img_a, img_b, flow_cd, img_c, img_d)

    def scale_immediate(self, flow_ab, img_a, img_b, scale):
        img_a_ph = tf.placeholder(shape=img_a.shape, dtype=tf.float32)
        img_b_ph = tf.placeholder(shape=img_b.shape, dtype=tf.float32)
        flowab_ph = tf.placeholder(shape=flow_ab.shape, dtype=tf.float32)
        flow_scaled, images_scaled = tf_scale_flow(flowab_ph, [img_a_ph, img_b_ph], scale)
        img_a_scaled = images_scaled[0]
        img_b_scaled = images_scaled[1]
        feed_dict = {
            img_a_ph: img_a,
            img_b_ph: img_b,
            flowab_ph: flow_ab
        }
        return self.sess.run([flow_scaled, img_a_scaled, img_b_scaled], feed_dict=feed_dict)

    def flip_immediate(self, flow_ab, img_a, img_b, flip_left_right, flip_up_down):
        img_a_ph = tf.placeholder(shape=img_a.shape, dtype=tf.float32)
        img_b_ph = tf.placeholder(shape=img_b.shape, dtype=tf.float32)
        flowab_ph = tf.placeholder(shape=flow_ab.shape, dtype=tf.float32)
        flow_flipped, images_flipped = tf_flip_flow(flowab_ph, [img_a_ph, img_b_ph], flip_left_right, flip_up_down)
        img_a_flipped = images_flipped[0]
        img_b_flipped = images_flipped[1]
        feed_dict = {
            img_a_ph: img_a,
            img_b_ph: img_b,
            flowab_ph: flow_ab
        }
        return self.sess.run([flow_flipped, img_a_flipped, img_b_flipped], feed_dict=feed_dict)

    def run_case(self, flow_ab, img_a, img_b, flow_cd, img_c, img_d):
        """
        Helper test method.
        """
        mask_ab = np.ones(shape=img_a.shape)
        mask_cd = np.ones(shape=img_a.shape)

        H = img_a.shape[0]
        W = img_a.shape[1]
        C = img_a.shape[2]

        # Create the graph.
        img_shape = [None, H, W, C]
        flow_shape = [None, H, W, 2]
        input = tf.placeholder(shape=img_shape, dtype=tf.float32)
        flow_tensor = tf.placeholder(shape=flow_shape, dtype=tf.float32)
        warped_tensor = backward_warp(input, flow_tensor)

        # Run.
        feed_dict = {
            input: [img_b, img_d, mask_ab, mask_cd],
            flow_tensor: [flow_ab, flow_cd, flow_ab, flow_cd]
        }
        warped_image = self.sess.run(warped_tensor, feed_dict=feed_dict)
        # Get the masked errors.
        error_ab_img = np.abs(warped_image[0] - img_a) * warped_image[2]
        error_cd_img = np.abs(warped_image[1] - img_c) * warped_image[3]
        error_ab = np.mean(error_ab_img)
        error_cd = np.mean(error_cd_img)
        # Assert a < 1.3% average error.
        self.assertLess(error_ab, 0.013)
        self.assertLess(error_cd, 0.013)

        if SHOW_AUGMENTATION_DEBUG_IMAGES:
            show_image(warped_image[0])
            show_image(warped_image[1])
            show_image(error_ab_img)
            show_image(error_cd_img)


if __name__ == '__main__':
    unittest.main()
