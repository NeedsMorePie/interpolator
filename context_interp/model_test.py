import numpy as np
import tensorflow as tf
import unittest
from context_interp.model import ContextInterp
from tensorflow.contrib.layers import l2_regularizer


class TestContextInterp(unittest.TestCase):

    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

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

        # Currently a stop_gradient is applied to the input of GridNet.
        # _, warped_a_b_tensor, warped_b_a_tensor, _, _ = output_tensors
        # grad_tensors = tf.gradients(interpolated_tensor, [warped_a_b_tensor, warped_b_a_tensor])
        # for grad_tensor in grad_tensors:
        #     self.assertNotEqual(grad_tensor, None)

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
        grad_tensors = tf.gradients(loss_tensor, interpolated_tensor)
        grads = self.sess.run(grad_tensors, feed_dict={image_a_placeholder: image_a,
                                                   image_b_placeholder: image_b,
                                                   gt_placeholder: np.zeros(shape=image_a.shape)})
        for gradient in grads:
            self.assertNotEqual(np.sum(gradient), 0.0)

        self.assertNotAlmostEqual(loss, 0.0)
        tf.reset_default_graph()
