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
        image_a[:, 10:height - 10, 10:width - 10, :] = 1.0
        image_b = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        image_b[:, 5:height - 5, 5:width - 5, :] = 1.0
        dummy_flow = np.ones(shape=[batch_size, height, width, 2], dtype=np.float32)

        self.sess.run(tf.global_variables_initializer())

        query = [final_flow] + previous_flows
        results = self.sess.run(query, feed_dict={input_image_a: image_a, input_image_b: image_b})

        self.assertEqual(len(results), 7)

        # Test that the default values are working.
        self.assertTupleEqual(results[0].shape, (batch_size, height, width, 2))
        self.assertTupleEqual(results[1].shape, (batch_size, height/64, width/64, 2))
        self.assertTupleEqual(results[2].shape, (batch_size, height/32, width/32, 2))
        self.assertTupleEqual(results[3].shape, (batch_size, height/16, width/16, 2))
        self.assertTupleEqual(results[4].shape, (batch_size, height/8, width/8, 2))
        self.assertTupleEqual(results[5].shape, (batch_size, height/4, width/4, 2))
        self.assertTupleEqual(results[6].shape, (batch_size, height/4, width/4, 2))

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
        # Check the losses.
        loss_value = self.sess.run(training_loss, feed_dict={input_image_a: image_a, input_image_b: image_b,
                                                             gt_placeholder: dummy_flow})
        self.assertNotAlmostEqual(loss_value[0], 0.0)

        # Check the gradients.
        loss_grad_ops = tf.gradients(training_loss, trainable_vars + [input_image_a, input_image_b])
        self.assertGreater(len(loss_grad_ops), 0)
        for grad in loss_grad_ops:
            self.assertNotEqual(grad, None)
        grads = self.sess.run(loss_grad_ops, feed_dict={input_image_a: image_a, input_image_b: image_b,
                                                        gt_placeholder: dummy_flow})
        for grad in grads:
            self.assertNotAlmostEqual(0.0, np.sum(grad))

    def test_network_fine_tuning_loss(self):
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
        image_a[:, 10:height - 10, 10:width - 10, :] = 1.0
        image_b = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        image_b[:, 5:height - 5, 5:width - 5, :] = 1.0
        dummy_flow = np.ones(shape=[batch_size, height, width, 2], dtype=np.float32)

        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables(scope='pwc_net')

        # Check that the gradients are flowing.
        grad_op = tf.gradients(tf.reduce_mean(final_flow), trainable_vars + [input_image_a, input_image_b])
        for grad in grad_op:
            self.assertNotEqual(grad, None)

        # Get the losses.
        gt_placeholder = tf.placeholder(shape=[None, height, width, 2], dtype=tf.float32)
        training_loss = self.pwc_net.get_fine_tuning_loss(previous_flows, gt_placeholder)
        # Check the loss.
        loss_value = self.sess.run(training_loss, feed_dict={input_image_a: image_a, input_image_b: image_b,
                                                             gt_placeholder: dummy_flow})
        self.assertNotAlmostEqual(loss_value[0], 0.0)

        # Check the gradients.
        loss_grad_ops = tf.gradients(training_loss, trainable_vars + [input_image_a, input_image_b])
        self.assertGreater(len(loss_grad_ops), 0)
        for grad in loss_grad_ops:
            self.assertNotEqual(grad, None)
        grads = self.sess.run(loss_grad_ops, feed_dict={input_image_a: image_a, input_image_b: image_b,
                                                        gt_placeholder: dummy_flow})
        for grad in grads:
            self.assertNotAlmostEqual(0.0, np.sum(grad))

    def test_network_unflow_loss(self):
        """
        Sets up the network's bidirectional pass and ensures that all shapes are expected.
        """
        height = 128
        width = 128
        num_features = 3
        batch_size = 2

        # Create the graph.
        input_image_a = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        input_image_b = tf.placeholder(shape=[None, height, width, num_features], dtype=tf.float32)
        forward_flow, backward_flow, forward_flows, backward_flows = self.pwc_net.get_forward_bidirectional(
            input_image_a, input_image_b)
        self.assertEqual(6, len(forward_flows))
        self.assertEqual(6, len(backward_flows))

        image_a = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        image_a[:, 10:height - 10, 10:width - 10, :] = 1.0
        image_b = np.zeros(shape=[batch_size, height, width, num_features], dtype=np.float32)
        image_b[:, 5:height - 5, 5:width - 5, :] = 1.0

        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables(scope='pwc_net')

        # Test output shapes.
        outputs = self.sess.run([forward_flow, backward_flow] + forward_flows + backward_flows,
                                feed_dict={input_image_a: image_a, input_image_b: image_b})
        self.assertTupleEqual((batch_size, height, width, 2), outputs[0].shape)
        self.assertTupleEqual((batch_size, height, width, 2), outputs[1].shape)
        self.assertTupleEqual((batch_size, height / 64, width / 64, 2), outputs[2].shape)
        self.assertTupleEqual((batch_size, height / 32, width / 32, 2), outputs[3].shape)
        self.assertTupleEqual((batch_size, height / 16, width / 16, 2), outputs[4].shape)
        self.assertTupleEqual((batch_size, height / 8, width / 8, 2), outputs[5].shape)
        self.assertTupleEqual((batch_size, height / 4, width / 4, 2), outputs[6].shape)
        self.assertTupleEqual((batch_size, height / 4, width / 4, 2), outputs[7].shape)
        self.assertTupleEqual((batch_size, height / 64, width / 64, 2), outputs[8].shape)
        self.assertTupleEqual((batch_size, height / 32, width / 32, 2), outputs[9].shape)
        self.assertTupleEqual((batch_size, height / 16, width / 16, 2), outputs[10].shape)
        self.assertTupleEqual((batch_size, height / 8, width / 8, 2), outputs[11].shape)
        self.assertTupleEqual((batch_size, height / 4, width / 4, 2), outputs[12].shape)
        self.assertTupleEqual((batch_size, height / 4, width / 4, 2), outputs[13].shape)

        # Check that the gradients are flowing.
        grad_op = tf.gradients(forward_flow, trainable_vars + [input_image_a, input_image_b])
        for grad in grad_op:
            self.assertNotEqual(grad, None)
        grad_op = tf.gradients(backward_flow, trainable_vars + [input_image_a, input_image_b])
        for grad in grad_op:
            self.assertNotEqual(grad, None)

        # Get the loss.
        training_loss, _, _, _, _ = self.pwc_net.get_unflow_training_loss(input_image_a, input_image_b,
                                                                          forward_flows, backward_flows)
        # Check the loss.
        loss_value = self.sess.run(training_loss, feed_dict={input_image_a: image_a, input_image_b: image_b})
        self.assertNotAlmostEqual(loss_value, 0.0)

        # Check the gradients.
        loss_grad_ops = tf.gradients(training_loss, trainable_vars + [input_image_a, input_image_b])
        self.assertGreater(len(loss_grad_ops), 0)
        for grad in loss_grad_ops:
            self.assertNotEqual(grad, None)
        grads = self.sess.run(loss_grad_ops, feed_dict={input_image_a: image_a, input_image_b: image_b})
        for grad in grads:
            self.assertNotAlmostEqual(0.0, np.sum(grad))


if __name__ == '__main__':
    unittest.main()
