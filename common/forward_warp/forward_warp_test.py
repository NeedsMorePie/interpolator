import os
import unittest
import numpy as np
import tensorflow as tf
from common.utils.img import read_image, show_image
from common.forward_warp.forward_warp import forward_warp, create_disocclusion_mask, is_forward_warp_cuda
from common.utils.flow import read_flow_file
from tensorflow.python.ops import gradient_checker


VISUALIZE = False
WRITE_TO_VIDEO = False


if not is_forward_warp_cuda():
    class TestForwardWarp(unittest.TestCase):
        def setUp(self):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        def test_forward_warp_whole_1(self):
            height = 2
            width = 2

            # Flow is in (x, y) order.
            # Splats the top-left pixel right in the center.
            flow = [[
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]]
            ]]
            features = [[
                [[4, 0], [0, 0]],
                [[1, 1], [0, 0]]
            ]]
            expected_warp = [[
                [[0, 0], [0, 0]],
                [[1, 1], [4, 0]]
            ]]

            flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            warp_tensor = forward_warp(features_tensor, flow_tensor)
            warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            self.assertEqual(warp.tolist(), expected_warp)

        def test_forward_warp_partial_1(self):
            height = 2
            width = 2

            # Flow is in (x, y) order.
            # Splats the top-left pixel right in the center.
            flow = [[
                [[0.5, 0.5], [0, 0]],
                [[0, 0], [0, 0]]
            ]]
            features = [[
                [[4, 0], [0, 0]],
                [[1, 1], [0, 0]]
            ]]
            expected_warp = [[
                [[1, 0], [1, 0]],
                [[2, 1], [1, 0]]
            ]]

            flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            warp_tensor = forward_warp(features_tensor, flow_tensor)
            warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            self.assertEqual(warp.tolist(), expected_warp)

        def test_forward_warp_partial_2(self):
            height = 3
            width = 2

            # Flow is in (x, y) order.
            # Splats the top-left pixel right in the center.
            flow = [[
                [[0.5, 0.5], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [-0.5, -0.5]]
            ]]
            features = [[
                [[4, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [-4, -4]]
            ]]
            expected_warp = [[
                [[1, 0], [1, 0]],
                [[1, 0], [0, -1]],
                [[-1, -1], [-1, -1]]
            ]]

            flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            warp_tensor = forward_warp(features_tensor, flow_tensor)
            warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            self.assertEqual(warp.tolist(), expected_warp)

            # Check for gradients.
            grads_tensor = tf.gradients(warp_tensor[0][0][0], [flow_tensor, features_tensor])
            for grad_tensor in grads_tensor:
                self.assertNotEqual(grad_tensor, None)

            grads = self.sess.run(grads_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            flow_grads, feature_grads = grads[0][0], grads[1][0]
            self.assertNotEqual(np.sum(flow_grads), 0.0)
            self.assertNotEqual(np.sum(feature_grads), 0.0)

        def test_forward_warp_oob(self):
            """
            Note that oob == out of bounds.
            """
            height = 3
            width = 2

            # Flow is in (x, y) order.
            # Splats the top-left pixel right in the center.
            flow = [[
                [[1.5, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [-10, -10]]
            ]]
            features = [[
                [[4, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [-4, -4]]
            ]]
            expected_warp = [[
                [[0, 0], [2, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]]
            ]]

            flow_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            features_tensor = tf.placeholder(tf.float32, (1, height, width, 2))
            warp_tensor = forward_warp(features_tensor, flow_tensor)
            warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            self.assertEqual(warp.tolist(), expected_warp)

        def test_forward_warp_batch(self):
            height = 2
            width = 2

            # Flow is in (x, y) order.
            # Splats the top-left pixel right in the center.
            flow = [
                [
                    [[0.5, 0.5], [0, 0]],
                    [[0, 0], [0, 0]]
                ],
                [
                    [[1, 1], [0, 0]],
                    [[0, 0], [0, 0]]
                ]
            ]
            features = [
                [
                    [[4, 0], [0, 0]],
                    [[1, 1], [0, 0]]
                ],
                [
                    [[100, 0], [0, 0]],
                    [[1, 1], [0, 0]]
                ]
            ]
            expected_warp = [
                [
                    [[1, 0], [1, 0]],
                    [[2, 1], [1, 0]]
                ],
                [
                    [[0, 0], [0, 0]],
                    [[1, 1], [100, 0]]
                ]
            ]

            flow_tensor = tf.placeholder(tf.float32, (2, height, width, 2))
            features_tensor = tf.placeholder(tf.float32, (2, height, width, 2))
            warp_tensor = forward_warp(features_tensor, flow_tensor)
            warp = self.sess.run(warp_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            self.assertEqual(warp[0].tolist(), expected_warp[0])
            self.assertEqual(warp[1].tolist(), expected_warp[1])

            # Check for gradients.
            grads_tensor = tf.gradients(warp_tensor[0][0][0], [flow_tensor, features_tensor])
            for grad_tensor in grads_tensor:
                self.assertNotEqual(grad_tensor, None)

            grads = self.sess.run(grads_tensor, feed_dict={flow_tensor: flow, features_tensor: features})
            flow_grads, feature_grads = grads[0][0], grads[1][0]
            self.assertNotEqual(np.sum(flow_grads), 0.0)
            self.assertNotEqual(np.sum(feature_grads), 0.0)


class TestForwardWarpCommon(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.flow_path = os.path.join('pwcnet', 'warp', 'test_data', 'flow_ab.flo')
        self.image_path_a = os.path.join('pwcnet', 'warp', 'test_data', 'image_a.png')
        self.image_path_b = os.path.join('pwcnet', 'warp', 'test_data', 'image_b.png')

        self.max_allowable_grad_err = 5e-4

    def test_visualization(self):
        if not VISUALIZE:
            return

        flow_ab = [read_flow_file(self.flow_path)]
        img_a = [read_image(self.image_path_a, as_float=True)]
        t_tensor = tf.placeholder(tf.float32, None)
        flow_ab_tensor = tf.placeholder(tf.float32, np.shape(flow_ab))
        img_a_tensor = tf.placeholder(tf.float32, np.shape(img_a))
        warp_tensor = forward_warp(img_a_tensor, t_tensor * flow_ab_tensor)

        warp = self.sess.run(warp_tensor, feed_dict={flow_ab_tensor: flow_ab, img_a_tensor: img_a, t_tensor: 1.0})
        warp = np.clip(warp[0], 0.0, 1.0)
        try:
            show_image(warp)
        except:
            print('show_image(warp) failed.')

        # For writing to video.
        if WRITE_TO_VIDEO:
            if not os.path.exists('outputs'):
                os.makedirs('outputs')

            import cv2
            import matplotlib.image as mpimg
            height = img_a[0].shape[0]
            width = img_a[0].shape[1]
            writer = cv2.VideoWriter('outputs/warped.avi',
                                     cv2.VideoWriter_fourcc(*'MJPG'), 20, (width, height))
            steps = 60
            for i in range(steps):
                print('Writing video at step %d' % i)
                t = i * (1.0 / float(steps))
                warped = self.sess.run(warp_tensor,
                                       feed_dict={flow_ab_tensor: flow_ab, img_a_tensor: img_a, t_tensor: t})
                warped = warped[0]
                warped = np.clip(warped, 0.0, 1.0)
                output_path = 'outputs/out-%.2f.png' % t
                mpimg.imsave(output_path, warped)
                writer.write(cv2.imread(output_path))
            writer.release()

    def test_warp_error(self):
        flow_ab = [read_flow_file(self.flow_path)]
        img_a = [read_image(self.image_path_a, as_float=True)]
        img_b = read_image(self.image_path_b, as_float=True)
        flow_ab_tensor = tf.placeholder(tf.float32, np.shape(flow_ab))
        img_a_tensor = tf.placeholder(tf.float32, np.shape(img_a))
        warp_tensor = forward_warp(img_a_tensor, flow_ab_tensor, splat_variance=0.3)
        mask = 1.0 - create_disocclusion_mask(flow_ab_tensor)

        warp, mask = self.sess.run([warp_tensor, mask], feed_dict={flow_ab_tensor: flow_ab, img_a_tensor: img_a})
        warp = np.clip(warp[0], 0.0, 1.0)

        self.assertLess(np.average(np.abs(warp - img_b) * mask[0]), 0.0212)

    def test_create_disocclusion_map(self):
        height = 3
        width = 3

        flow_tensor = tf.placeholder(shape=(None, height, width, 2), dtype=tf.float32)
        mask_tensor = create_disocclusion_mask(flow_tensor, splat_variance=0.2)

        flow = np.asarray([
            [
                [[1., 1.], [1., 1.], [0., 0.]],
                [[1., 1.], [0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.], [-1., -1.]]
            ]
        ], dtype=np.float32)

        mask = self.sess.run(mask_tensor, feed_dict={flow_tensor: flow})

        expected_mask = np.asarray([
            [
                [[1.], [1.], [0.]],
                [[1.], [0.], [0.]],
                [[0.], [0.], [1.]]
            ]
        ], dtype=np.float32)
        self.assertTrue(np.allclose(expected_mask, mask))

    def test_create_disocclusion_map_batched(self):
        height = 3
        width = 3

        flow_tensor = tf.placeholder(shape=(None, height, width, 2), dtype=tf.float32)
        mask_tensor = create_disocclusion_mask(flow_tensor, splat_variance=0.2)

        flow = np.asarray([
            [
                [[2., 2.], [0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.], [0., 0.]]
            ],
            [
                [[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.], [-2., -2.]]
            ]
        ], dtype=np.float32)

        mask = self.sess.run(mask_tensor, feed_dict={flow_tensor: flow})

        expected_mask = np.asarray([
            [
                [[1.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]]
            ],
            [
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [1.]]
            ]
        ], dtype=np.float32)
        self.assertTrue(np.allclose(expected_mask, mask))

    def test_create_disocclusion_map_no_gradient(self):
        height = 3
        width = 3
        batch_size = 2

        flow_tensor = tf.placeholder(shape=(batch_size, height, width, 2), dtype=tf.float32)
        mask_tensor = create_disocclusion_mask(flow_tensor)
        grad = tf.gradients(mask_tensor, flow_tensor)[0]
        self.assertEqual(None, grad)

    def test_gradients_errors(self):
        self.gradient_errors_helper(splat_variance=1.0)

    def test_gradients_errors_low_splat(self):
        if not is_forward_warp_cuda():
            return
        self.gradient_errors_helper(splat_variance=0.2)

    def test_gradients_errors_high_splat(self):
        if not is_forward_warp_cuda():
            return
        self.gradient_errors_helper(splat_variance=0.4)

    def gradient_errors_helper(self, splat_variance):
        with self.sess:
            # This test is flaky, so retry if fail.
            num_tries = 2
            error1 = 0
            error2 = 0
            for i in range(num_tries):
                img_shape = (16, 3, 4, 4)
                flow_shape = (16, 3, 4, 2)
                img_a = np.random.rand(*img_shape)
                flow_ab = (np.random.rand(*flow_shape) - 0.5) * 3
                input = tf.placeholder(shape=img_a.shape, dtype=tf.float32)
                flow_tensor = tf.placeholder(shape=flow_ab.shape, dtype=tf.float32)
                warped_tensor = forward_warp(input, flow_tensor, splat_variance=splat_variance)

                error1 = gradient_checker.compute_gradient_error(input, img_a.shape, warped_tensor, img_a.shape,
                                                                 extra_feed_dict={flow_tensor: flow_ab},
                                                                 x_init_value=img_a)
                error2 = gradient_checker.compute_gradient_error(flow_tensor, flow_ab.shape, warped_tensor,
                                                                 img_a.shape, extra_feed_dict={input: img_a},
                                                                 x_init_value=flow_ab)
                if error1 <= self.max_allowable_grad_err and error2 <= self.max_allowable_grad_err:
                    return
            self.assertLessEqual(max(error1, error2), self.max_allowable_grad_err,
                                 'Exceeded the error threshold. Note that this test may be flaky.')


if __name__ == '__main__':
    unittest.main()
