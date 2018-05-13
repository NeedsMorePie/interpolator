import unittest

import numpy as np

from pwcnet.warp.warp import *
from utils.flow import read_flow_file
from utils.img import read_image, show_image

SHOW_WARPED_IMAGES = False


class TestSpacialTransformTranslate(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_single_transform(self):
        """
        This test creates a box and translates it in 2 ways:
        1) Regular translation via numpy.
        2) Translation via SpacialTransformNetwork.
        Then, it diffs both translation methods and asserts that they are close.
        """
        # Image dimensions.
        height = 70
        width = 90
        channels = 3
        image_shape = [height, width, channels]

        # Defines how many pixels to translate by.
        x_trans_amount = 20
        y_trans_amount = 10

        # Defines the square position and size.
        y_start = 10
        y_end = 40
        x_start = 20
        x_end = 50

        # The maximum diff is the circumference of the square.
        max_diff = (x_end - x_start) * 2 + (y_end - y_start) * 2

        # Input.
        box_image = np.zeros(shape=image_shape, dtype=np.float)
        box_image[y_start:y_end, x_start:x_end, :] = 1.0

        # Expected output.
        translated_box_image = np.zeros(shape=image_shape, dtype=np.float)
        translated_box_image[y_start + y_trans_amount:y_end + y_trans_amount,
                             x_start + x_trans_amount:x_end + x_trans_amount, :] = 1.0

        # Warp matrix to achieve the expected output.
        translation = np.asarray([[[[x_trans_amount, y_trans_amount]]]])
        translation = np.tile(translation, [1, height, width, 1])
        warp = optical_flow_to_transforms_immediate(translation, self.sess)[0, 0, 0, :]

        # Create the graph and run it to get the actual output.
        input = tf.placeholder(shape=[None] + image_shape, dtype=tf.float32)
        theta = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        transformed = spatial_transformer_network(input, theta)
        # Run with batch size of 2.
        transformed_image = self.sess.run(transformed, feed_dict={input: [translated_box_image, translated_box_image],
                                                                  theta: [warp, warp]})

        # Do the diff.
        diff_img = np.abs(transformed_image[0] - box_image)
        diff = np.sum(np.mean(diff_img, axis=-1))
        self.assertLess(diff, max_diff)
        diff_img = np.abs(transformed_image[1] - box_image)
        diff = np.sum(np.mean(diff_img, axis=-1))
        self.assertLess(diff, max_diff)

        if SHOW_WARPED_IMAGES:
            show_image(transformed_image[1])
            show_image(diff_img)

    def test_flow_to_matrix(self):
        """
        Tests that we can take an optical flow image and convert it into flat 2D transform matrices.
        """
        flow = np.asarray([[[1, 1], [2, 2]],
                           [[3, 3], [4, 4]],
                           [[5, 5], [6, 6]]], dtype=np.float32)
        flow = np.stack([flow, flow])  # Create a batch size of 2.
        flow_holder = tf.placeholder(shape=flow.shape, dtype=tf.float32)
        transforms_tensor = optical_flow_to_transforms(flow_holder)
        transforms = self.sess.run(transforms_tensor, feed_dict={flow_holder: flow})

        height_scale = 2.0 / float(flow.shape[1])
        width_scale = 2.0 / float(flow.shape[2])
        expected_transforms = np.asarray([[[1 * width_scale, 1 * height_scale], [2 * width_scale, 2 * height_scale]],
                                          [[3 * width_scale, 3 * height_scale], [4 * width_scale, 4 * height_scale]],
                                          [[5 * width_scale, 5 * height_scale], [6 * width_scale, 6 * height_scale]]],
                                         dtype=np.float32)
        expected_transforms = np.stack([expected_transforms, expected_transforms])
        self.assertTrue(np.allclose(transforms, expected_transforms))

    def test_optical_flow_warp(self):
        """
        Runs warp test against Sintel ground truth and checks for <2% average pixel error.
        Also masks out the occluded warp regions when computing the error.
        """
        # Load from files.
        flow_ab = read_flow_file('pwcnet/warp/test_data/flow_ab.flo')
        img_a = read_image('pwcnet/warp/test_data/image_a.png', as_float=True)
        img_b = read_image('pwcnet/warp/test_data/image_b.png', as_float=True)
        flow_cd = read_flow_file('pwcnet/warp/test_data/flow_cd.flo')
        img_c = read_image('pwcnet/warp/test_data/image_c.png', as_float=True)
        img_d = read_image('pwcnet/warp/test_data/image_d.png', as_float=True)
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
        warped_tensor = warp_via_flow(input, flow_tensor)

        # Run.
        warped_image = self.sess.run(warped_tensor, feed_dict={input: [img_b, img_d, mask_ab, mask_cd],
                                                               flow_tensor: [flow_ab, flow_cd, flow_ab, flow_cd]})
        # Get the masked errors.
        error_ab_img = np.abs(warped_image[0] - img_a) * warped_image[2]
        error_cd_img = np.abs(warped_image[1] - img_c) * warped_image[3]
        error_ab = np.mean(error_ab_img)
        error_cd = np.mean(error_cd_img)
        # Assert a < 1.6% average error.
        self.assertLess(error_ab, 0.016)
        self.assertLess(error_cd, 0.016)

        if SHOW_WARPED_IMAGES:
            show_image(warped_image[0])
            show_image(warped_image[1])
            show_image(error_ab_img)
            show_image(error_cd_img)

    def test_gradients(self):
        """
        Test to see if the gradients flow.
        """
        # Load from files.
        flow_ab = read_flow_file('pwcnet/warp/test_data/flow_ab.flo')
        img_b = read_image('pwcnet/warp/test_data/image_b.png', as_float=True)

        H = img_b.shape[0]
        W = img_b.shape[1]
        C = img_b.shape[2]
        img_shape = [None, H, W, C]
        flow_shape = [None, H, W, 2]
        input = tf.placeholder(shape=img_shape, dtype=tf.float32)
        flow_tensor = tf.placeholder(shape=flow_shape, dtype=tf.float32)
        warped_tensor = warp_via_flow(input, flow_tensor)

        dummy_loss = tf.reduce_mean(warped_tensor)
        grad_op = tf.gradients(dummy_loss, [input, flow_tensor])
        grads = self.sess.run(grad_op, feed_dict={input: [img_b], flow_tensor: [flow_ab]})
        for gradient in grads:
            self.assertNotEqual(np.sum(gradient), 0.0)


if __name__ == '__main__':
    unittest.main()
