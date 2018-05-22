import os.path
import tensorflow as tf
from data.flow.flowdata import FlowDataSet
from pwcnet.model import PWCNet
from train.trainer import Trainer


# TODO: Tensorboard.
class PWCNetTrainer(Trainer):
    def __init__(self, model, dataset, session, config, verbose=True):
        super().__init__(model, dataset, session, config, verbose)
        assert isinstance(self.model, PWCNet)
        assert isinstance(dataset, FlowDataSet)

        self.dataset.load(self.session)
        images_a, images_b, flows = self.dataset.get_next_batch()

        # Get the train network.
        final_flow, previous_flows = self.model.get_forward(images_a, images_b, reuse_variables=tf.AUTO_REUSE)
        self.loss, self.layer_losses = self.model.get_training_loss(previous_flows, flows)

        # Get the optimizer.
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            self.train_op = tf.train.AdamOptimizer(config['learning_rate']).minimize(
                self.loss, global_step=self.global_step)

        # Checkpoint saving.
        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(os.path.join(config['checkpoint_directory'], 'valid'),
                                            self.session.graph)

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
        avg_loss = 0.0
        iterations = 0
        while True:
            try:
                loss = self.session.run(self.loss, feed_dict=self.dataset.get_validation_feed_dict())
            except tf.errors.OutOfRangeError:
                # End of validation epoch.
                break
            avg_loss += loss
            iterations += 1
        avg_loss /= float(iterations)

    def _eval_global_step(self):
        return self.session.run(self.global_step)
