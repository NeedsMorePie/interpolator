import os.path
import tensorflow as tf
from data.interp.interp_data import InterpDataSet
from context_interp.model import ContextInterp
from train.trainer import Trainer
from utils.misc import print_progress_bar
from utils.flow import get_tf_flow_visualization
from utils.tf import AdamaxOptimizer, optimistic_restore, Logger


class ContextInterpTrainer(Trainer):
    def __init__(self, model, dataset, session, config, verbose=True):
        super().__init__(model, dataset, session, config, verbose)
        assert isinstance(self.model, ContextInterp)
        assert isinstance(dataset, InterpDataSet)

        self.dataset.load(self.session)
        self.next_sequence_tensor, self.next_sequence_timing_tensor = self.dataset.get_next_batch()

        self.images_a = self.next_sequence_tensor[:, 0]
        self.images_b = self.next_sequence_tensor[:, 1]
        self.images_c = self.next_sequence_tensor[:, 2]

        # Get the train network.
        model_outputs = self.model.get_forward(self.images_a, self.images_c, 0.5, reuse_variables=tf.AUTO_REUSE)
        self.images_b_pred, self.warped_a_c, self.warped_c_a, self.flow_a_c, self.flow_c_a = model_outputs
        if self.config['fine_tune']:
            self.loss = self.model.get_fine_tuning_loss(self.images_b_pred, self.images_b)
        else:
            self.loss = self.model.get_training_loss(self.images_b_pred, self.images_b)

        # Get the optimizer.
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            self.train_op = AdamaxOptimizer(config['learning_rate'], beta1=0.9, beta2=0.999).minimize(
                self.loss, global_step=self.global_step
            )

        # Checkpoint saving.
        self.saver = tf.train.Saver()
        self.valid_logger = Logger(os.path.join(self.config['checkpoint_directory'], 'valid'))
        self.train_logger = Logger(os.path.join(self.config['checkpoint_directory'], 'train'))
        self._make_summaries_dict()

    def restore(self):
        """
        Overridden.
        """
        if tf.train.latest_checkpoint(self.config['checkpoint_directory']) is not None:
            print('Restoring checkpoint...')
            checkpoint_file = tf.train.latest_checkpoint(self.config['checkpoint_directory'])
            optimistic_restore(self.session, checkpoint_file)

    def train_for(self, iterations):
        """
        Overridden.
        """
        for i in range(iterations):
            if i == iterations - 1:
                # Write the summary on the last iteration.
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = self.session.run(self.train_op,
                                           feed_dict=self.dataset.get_train_feed_dict(),
                                           options=run_options, run_metadata=run_metadata)
                global_step = self._eval_global_step()

                # Write summaries.
                loss_key = 'total_loss'
                np_summary_dict = self._get_np_summaries(feed_dict=self.dataset.get_train_feed_dict())
                for key, value in np_summary_dict.items():
                    if key != loss_key:
                        self.train_logger.log_images(key, value, global_step)
                self.train_logger.log_scalar(loss_key, np_summary_dict[loss_key], global_step)
                self.train_logger.add_run_metadata(run_metadata, global_step)
            else:
                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=self.dataset.get_train_feed_dict())
            print_progress_bar(i + 1, iterations, prefix='Train Steps', suffix='Complete', use_percentage=False)

        print('Saving model checkpoint...')
        save_path = self.saver.save(self.session, os.path.join(self.config['checkpoint_directory'], 'model.ckpt'),
                                    global_step=self._eval_global_step())
        print('Model saved in path:', save_path)

    def validate(self):
        """
        Overridden.
        """
        self.dataset.init_validation_data(self.session)
        global_step = self._eval_global_step()
        loss_key = 'total_loss'
        iter = 0
        total_loss = 0
        image_logging_period = 100
        while True:
            try:
                np_summary_dict = self._get_np_summaries(self.dataset.get_validation_feed_dict())
                if iter % image_logging_period == 0:
                    for key, value in np_summary_dict.items():
                        if key != loss_key:
                            self.valid_logger.log_images(key, value, global_step)
                total_loss += np_summary_dict[loss_key]
                iter += 1
            except tf.errors.OutOfRangeError:
                # End of validation epoch.
                break

        total_loss /= len(np_summary_dict)
        self.valid_logger.log_scalar(loss_key, total_loss, global_step)

    def _eval_global_step(self):
        return self.session.run(self.global_step)

    def _get_np_summaries(self, feed_dict):
        np_summaries_dict = {}
        tensors = []
        for key, value in self.summaries_dict.items():
            tensors.append(value)

        np_vals = self.session.run(tensors, feed_dict=feed_dict)
        for i, (key, value) in enumerate(self.summaries_dict.items()):
            np_summaries_dict[key] = np_vals[i]
        return np_summaries_dict

    def _make_summaries_dict(self):
        with tf.name_scope('summaries'):

            # Get overlays.
            weight = 0.60
            image_overlays = self.images_a * weight + self.images_c * (1 - weight)

            # Images will appear 'washed out' otherwise, due to tensorboard's own normalization scheme.
            clipped_preds = tf.clip_by_value(self.images_b_pred, 0.0, 1.0)
            clipped_warp_a_b = tf.clip_by_value(self.warped_a_c[..., 0:3], 0.0, 1.0)
            clipped_warp_b_a = tf.clip_by_value(self.warped_c_a[..., 0:3], 0.0, 1.0)

            self.summaries_dict = {
                'overlay': image_overlays,
                'total_loss': self.loss,
                'inbetween_gt': self.images_b,
                'inbetween_pred': clipped_preds,
                'warped_a_c': clipped_warp_a_b,
                'warped_c_a': clipped_warp_b_a,
                'flow_a_c': get_tf_flow_visualization(self.flow_a_c),
                'flow_c_a': get_tf_flow_visualization(self.flow_c_a)
            }
