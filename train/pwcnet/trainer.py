import os.path
import tensorflow as tf
from data.flow.flowdata import FlowDataSet
from pwcnet.model import PWCNet
from train.trainer import Trainer
from utils.flow import get_tf_flow_visualization


class PWCNetTrainer(Trainer):
    def __init__(self, model, dataset, session, config, verbose=True):
        super().__init__(model, dataset, session, config, verbose)
        assert isinstance(self.model, PWCNet)
        assert isinstance(dataset, FlowDataSet)

        self.dataset.load(self.session)
        self.images_a, self.images_b, self.flows = self.dataset.get_next_batch()

        # Get the train network.
        self.final_flow, self.previous_flows = self.model.get_forward(self.images_a, self.images_b,
                                                                      reuse_variables=tf.AUTO_REUSE)
        self.loss, self.layer_losses = self.model.get_training_loss(self.previous_flows, self.flows)

        # Get the optimizer.
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            self.train_op = tf.train.AdamOptimizer(config['learning_rate']).minimize(
                self.loss, global_step=self.global_step)

        # Checkpoint saving.
        self.saver = tf.train.Saver()

        self.merged_summ = None
        self.train_writer = None
        self.valid_writer = None
        self._make_summaries()

    def restore(self):
        """
        Overridden.
        """
        if tf.train.latest_checkpoint(self.config['checkpoint_directory']) is not None:
            print('Restoring checkpoint...')
            self.saver.restore(self.session, os.path.join(self.config['checkpoint_directory'], 'model.ckpt'))

    def train_for(self, iterations):
        """
        Overridden.
        """
        avg_loss = 0.0
        for i in range(iterations):
            if i == iterations - 1:
                # Write the summary on the last iteration.
                loss, _, summ = self.session.run([self.loss, self.train_op, self.merged_summ],
                                                 feed_dict=self.dataset.get_train_feed_dict())
                self.train_writer.add_summary(summ, self._eval_global_step())
            else:
                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=self.dataset.get_train_feed_dict())
            avg_loss += loss
        avg_loss /= float(iterations)

        print('Saving model checkpoint...')
        save_path = self.saver.save(self.session, os.path.join(self.config['checkpoint_directory'], 'model.ckpt'),
                                    global_step=self._eval_global_step())
        print('Model saved in path:', save_path)

    def validate(self):
        """
        Overridden.
        """
        self.dataset.init_validation_data(self.session)
        while True:
            try:
                summ = self.session.run(self.merged_summ, feed_dict=self.dataset.get_validation_feed_dict())
                self.valid_writer.add_summary(summ, self._eval_global_step())
            except tf.errors.OutOfRangeError:
                # End of validation epoch.
                break

    def _eval_global_step(self):
        return self.session.run(self.global_step)

    def _make_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('total_loss', self.loss)
            for i, layer_loss in enumerate(self.layer_losses):
                tf.summary.scalar('layer_' + str(i) + '_loss', layer_loss)
            for i, previous_flow in enumerate(self.previous_flows):
                tf.summary.image('flow_' + str(i), get_tf_flow_visualization(previous_flow))
            tf.summary.image('image_a', self.images_a)
            tf.summary.image('image_b', self.images_b)
            tf.summary.image('final_flow', get_tf_flow_visualization(self.final_flow))
            tf.summary.image('gt_flow', get_tf_flow_visualization(self.flows))

            self.merged_summ = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(os.path.join(self.config['checkpoint_directory'], 'train'),
                                                      self.session.graph)
            self.valid_writer = tf.summary.FileWriter(os.path.join(self.config['checkpoint_directory'], 'valid'))
