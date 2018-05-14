import numpy as np
import tensorflow as tf
import unittest
from pwcnet.model import PWCNet
from tensorflow.contrib.layers import l2_regularizer


class TestPWCModel(unittest.TestCase):
    def setUp(self):
        self.pwc_net = PWCNet(regularizer=l2_regularizer(1e-4))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_network(self):
        """
        Sets up the network's forward pass and ensures that all shapes are expected.
        """
        height = 128
        width = 128
        num_features = 3
        batch_size = 2

        # Create the graph.
        input_image_a = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        input_image_b = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        final_flow, previous_flows = self.pwc_net.get_forward(input_image_a, input_image_b)

        image_a = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        image_a[:, 10:height-10, 10:width-10, :] = 1.0
        image_b = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        image_b[:, 5:height - 5, 5:width - 5, :] = 1.0
        dummy_flow = np.ones(shape=[batch_size, height, width, 2], dtype=np.float32)

        self.sess.run(tf.global_variables_initializer())

        query = [final_flow] + previous_flows
        results = self.sess.run(query, feed_dict={input_image_a: image_a, input_image_b: image_b})

        self.assertEqual(len(results), 7)

        # Test that the default values are working.
        self.assertTrue(np.allclose(results[1].shape, np.asarray([batch_size, height/64, width/64, 2])))
        self.assertTrue(np.allclose(results[2].shape, np.asarray([batch_size, height/32, width/32, 2])))
        self.assertTrue(np.allclose(results[3].shape, np.asarray([batch_size, height/16, width/16, 2])))
        self.assertTrue(np.allclose(results[4].shape, np.asarray([batch_size, height/8, width/8, 2])))
        self.assertTrue(np.allclose(results[5].shape, np.asarray([batch_size, height/4, width/4, 2])))
        self.assertTrue(np.allclose(results[6].shape, np.asarray([batch_size, height/4, width/4, 2])))

        for i in range(1, 7):
            self.assertNotEqual(np.sum(results[i]), 0.0)

        trainable_vars = tf.trainable_variables(scope='pwc_net')

        # Check that the gradients are flowing.
        grad_op = tf.gradients(tf.reduce_mean(final_flow), trainable_vars + [input_image_a, input_image_b])
        for grad in grad_op:
            self.assertNotEqual(grad, None)

        # Get the losses.
        gt_placeholder = tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32)
        training_loss = self.pwc_net.get_training_loss(previous_flows, gt_placeholder)
        # Check the gradients with the loss.
        loss_grad_op = tf.gradients(training_loss, trainable_vars + [input_image_a, input_image_b])
        for grad in loss_grad_op:
            self.assertNotEqual(grad, None)
        loss_value = self.sess.run(training_loss, feed_dict={input_image_a: image_a, input_image_b: image_b,
                                                             gt_placeholder: dummy_flow})
        self.assertNotAlmostEqual(loss_value[0], 0.0)


if __name__ == '__main__':
    unittest.main()
