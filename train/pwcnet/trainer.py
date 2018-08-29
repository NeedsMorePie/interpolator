import os.path
import tensorflow as tf
from common.utils.flow import get_tf_flow_visualization
from common.utils.multi_gpu import get_available_gpus, TensorIO, create_train_op
from common.utils.profile import save_timeline
from data.flow.flow_data import FlowDataSet
from pwcnet.model import PWCNet
from train.trainer import Trainer


class PWCNetTrainer(Trainer):
    def __init__(self, model, dataset, session, config, verbose=True):
        super().__init__(model, dataset, session, config, verbose)
        assert isinstance(self.model, PWCNet)
        assert isinstance(dataset, FlowDataSet)

        self.dataset.load(self.session)
        self.images_a, self.images_b, self.flows = self.dataset.get_next_batch()

        # Create the network's forward and train ops.
        self.final_flow = None
        self.previous_flows = None
        self.loss = None
        self.layer_losses = None
        self.global_step = None
        self.train_op = None
        self._create_ops()

        # Summary variables.
        self.merged_summ = None
        self.train_writer = None
        self.valid_writer = None
        self.train_log_dir = os.path.join(self.config['checkpoint_directory'], 'train')
        self.valid_log_dir = os.path.join(self.config['checkpoint_directory'], 'valid')
        self._make_summaries()

        # Checkpoint saving.
        self.saver = tf.train.Saver()
        self.npz_save_file = os.path.join(self.config['checkpoint_directory'], 'pwcnet_weights.npz')

    def _create_ops(self):
        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])

        def build_network_outputs(images_a, images_b, flows):
            # Get the train network.
            final_flow, previous_flows = self.model.get_forward(images_a, images_b, reuse_variables=tf.AUTO_REUSE)
            if self.config['fine_tune']:
                loss, layer_losses = self.model.get_fine_tuning_loss(previous_flows, flows)
            else:
                loss, layer_losses = self.model.get_training_loss(previous_flows, flows)

            return {
                'loss': TensorIO([loss], TensorIO.AVERAGED_SCALAR),
                'layer_losses': TensorIO(layer_losses, TensorIO.AVERAGED_SCALAR),
                'previous_flows': TensorIO(previous_flows, TensorIO.BATCH),
                'final_flow': TensorIO([final_flow], TensorIO.BATCH)
            }

        # Use a helper function to split the batch across multiple GPUs.
        self.train_op, self.global_step, outputs = create_train_op(
            optimizer, build_network_outputs, [self.images_a, self.images_b, self.flows], [],
            available_devices=get_available_gpus(), verbose=True)

        self.final_flow = outputs['final_flow'].first()
        self.previous_flows = outputs['previous_flows'].tensors
        self.loss = outputs['loss'].first()
        self.layer_losses = outputs['layer_losses'].tensors

    def restore(self):
        """
        Overridden.
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.train_log_dir)
        if latest_checkpoint is not None:
            print('Restoring checkpoint...')
            self.saver.restore(self.session, latest_checkpoint)
        if os.path.isfile(self.npz_save_file):
            print('Restoring weights from npz...')
            self.model.restore_from(self.npz_save_file, self.session)

    def train_for(self, iterations):
        """
        Overridden.
        """
        for i in range(iterations):
            if i == iterations - 1:
                # Write the summary on the last iteration.
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, summ = self.session.run([self.train_op, self.merged_summ],
                                           feed_dict=self.dataset.get_train_feed_dict(),
                                           options=run_options, run_metadata=run_metadata)
                global_step = self._eval_global_step()
                self.train_writer.add_run_metadata(run_metadata, 'step%d' % global_step, global_step=global_step)
                self.train_writer.add_summary(summ, global_step=global_step)
                self.train_writer.flush()
                self.model.save_to(self.npz_save_file, self.session)
                save_timeline(run_metadata, self.train_log_dir, detailed=False)
            else:
                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=self.dataset.get_train_feed_dict())

        print('Saving model checkpoint...')
        save_path = self.saver.save(self.session, os.path.join(self.train_log_dir, 'model.ckpt'),
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
        self.valid_writer.flush()

    def _eval_global_step(self):
        return self.session.run(self.global_step)

    def _make_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('total_loss', self.loss)
            for i, layer_loss in enumerate(self.layer_losses):
                tf.summary.scalar('layer_' + str(i) + '_loss', layer_loss)
            for i, previous_flow in enumerate(self.previous_flows):
                tf.summary.image('flow_' + str(i), get_tf_flow_visualization(previous_flow))
            tf.summary.image('image_a', tf.clip_by_value(self.images_a, 0.0, 1.0))
            tf.summary.image('image_b', tf.clip_by_value(self.images_b, 0.0, 1.0))
            tf.summary.image('final_flow', get_tf_flow_visualization(self.final_flow))
            tf.summary.image('gt_flow', get_tf_flow_visualization(self.flows))
            self.merged_summ = tf.summary.merge_all()

            # Config summary.
            text = []
            for key in sorted(self.config.keys()):
                text.append([key, str(self.config[key])])
            config_summary = tf.summary.text('Configurations', tf.convert_to_tensor(text))

            self.train_writer = tf.summary.FileWriter(self.train_log_dir, self.session.graph)
            self.valid_writer = tf.summary.FileWriter(self.valid_log_dir)

        # Write the config summary.
        self.train_writer.add_summary(self.session.run(config_summary), global_step=0)
        self.train_writer.flush()
