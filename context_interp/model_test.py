import numpy as np
import tensorflow as tf
import unittest
from context_interp.model import ContextInterp


class TestContextInterp(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.do_grads_flow_to_input = False

    def test_network(self):
        height = 128
        width = 64
        im_channels = 3
        batch_size = 2

        # Create the graph.
        model = ContextInterp()
        image_a_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
        image_b_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
        gt_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
        output_tensors = model.get_forward(image_a_placeholder, image_b_placeholder, 0.5)
        interpolated_tensor = output_tensors[0]

        image_a = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
        image_b = np.zeros(shape=[batch_size, height, width, im_channels], dtype=np.float32)
        image_a[:, 2:height-2, 2:width-2, :] = 1.0
        image_b[:, 4:height-4, 5:width-5, :] = 1.0

        loss_tensor = model.get_training_loss(interpolated_tensor, gt_placeholder)
        _, warped_a_b_tensor, warped_b_a_tensor, _, _ = output_tensors

        # Currently a stop_gradient is applied to the input of GridNet.
        if self.do_grads_flow_to_input:
            grad_tensors = tf.gradients(interpolated_tensor, [image_a_placeholder, image_b_placeholder])
            for grad_tensor in grad_tensors:
                self.assertNotEqual(grad_tensor, None)

        query = [loss_tensor] + list(output_tensors)
        self.sess.run(tf.global_variables_initializer())
        outputs_np = self.sess.run(query, feed_dict={image_a_placeholder: image_a,
                                                     image_b_placeholder: image_b,
                                                     gt_placeholder: np.zeros(shape=image_a.shape)})

        loss, interpolated, warped_a_b, warped_b_a, flow_a_b, flow_b_a = outputs_np
        warped_im_a_b = warped_a_b[..., :3]
        warped_feat_a_b = warped_a_b[..., 3:]
        warped_im_b_a = warped_b_a[..., :3]
        warped_feat_b_a = warped_b_a[..., 3:]

        # Check shapes.
        self.assertTrue(np.allclose(interpolated.shape, np.asarray([batch_size, height, width, im_channels])))
        self.assertTrue(np.allclose(warped_im_a_b.shape, np.asarray([batch_size, height, width, im_channels])))
        self.assertTrue(np.allclose(warped_im_b_a.shape, np.asarray([batch_size, height, width, im_channels])))
        self.assertTrue(np.allclose(warped_feat_a_b.shape[:-1], np.asarray([batch_size, height, width])))
        self.assertTrue(np.allclose(warped_feat_b_a.shape[:-1], np.asarray([batch_size, height, width])))
        self.assertTrue(np.allclose(warped_feat_a_b.shape, warped_feat_b_a.shape))

        # Check the gradients with the loss.
        grad_tensors = tf.gradients(loss_tensor, [interpolated_tensor, warped_a_b_tensor, warped_b_a_tensor])
        grads = self.sess.run(grad_tensors, feed_dict={image_a_placeholder: image_a,
                                                   image_b_placeholder: image_b,
                                                   gt_placeholder: np.zeros(shape=image_a.shape)})
        for gradient in grads:
            self.assertNotEqual(np.sum(gradient), 0.0)

        self.assertNotAlmostEqual(loss, 0.0)

    def test_shared_weights(self):
        # TODO: Enable this test after the issue is fixed.
        return

        height = 128
        width = 64
        im_channels = 3

        # Create the graph.
        model = ContextInterp(name='context_interp_shared')
        image_a_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
        image_b_placeholder = tf.placeholder(shape=[None, height, width, im_channels], dtype=tf.float32)
        trainable_vars_before = len(tf.trainable_variables())  # Global scope used to make sure we don't miss anything.
        model.get_forward(image_a_placeholder, image_b_placeholder, 0.5)
        trainable_vars_after = len(tf.trainable_variables())
        self.assertGreater(trainable_vars_after, trainable_vars_before)

        before = set(tf.trainable_variables())

        # Do it again and check that the number of trainable variables has not increased.
        model.get_forward(image_a_placeholder, image_b_placeholder, 0.5)
        diff = set(tf.trainable_variables()) - before
        for item in diff:
            print(item.name)
        self.assertEqual(trainable_vars_after, len(tf.trainable_variables()))


if __name__ == '__main__':
    unittest.main()
