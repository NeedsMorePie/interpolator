import os.path
import tensorflow as tf
from common.utils.flow import get_tf_flow_visualization
from common.utils.multi_gpu import average_gradients, get_available_gpus, accumulate_list_into
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
        available_gpus = get_available_gpus()
        if len(available_gpus) == 1:
            self._create_single_gpu_ops()
        else:
            self._create_multi_gpu_ops(available_gpus)

    def _create_single_gpu_ops(self):
        if self.verbose:
            print('Initializing single-gpu training...')
        # Get the train network.
        self.final_flow, self.previous_flows = self.model.get_forward(self.images_a, self.images_b,
                                                                      reuse_variables=tf.AUTO_REUSE)
        if self.config['fine_tune']:
            self.loss, self.layer_losses = self.model.get_fine_tuning_loss(self.previous_flows, self.flows)
        else:
            self.loss, self.layer_losses = self.model.get_training_loss(self.previous_flows, self.flows)

        # Get the optimizer.
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            self.train_op = tf.train.AdamOptimizer(self.config['learning_rate']).minimize(
                self.loss, global_step=self.global_step)

    def _create_multi_gpu_ops(self, available_gpus):
        """
        :param available_gpus: List of strings. I.e. ['/device:GPU:0', '/device:GPU:1'].
        :return: Nothing.
        """
        if self.verbose:
            print('Detected', available_gpus, 'GPUs. Initializing multi-gpu training...')

        with tf.name_scope('examples_per_gpu'):
            batch_size = tf.shape(self.images_a)[0]
            num_gpus = len(available_gpus)
            examples_per_gpu = tf.cast(batch_size / num_gpus, dtype=tf.int32)

        # Accumulation variables.
        final_flow_list, previous_flows_list = [], []
        loss_list, layer_losses_list = [], []
        tower_grads_and_vars = []

        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        for i, gpu in enumerate(available_gpus):
            if self.verbose:
                print('Creating tower for', gpu)

            with tf.name_scope('batch_distribution'):
                start = tf.cast(examples_per_gpu * i, dtype=tf.int32)
                end = tf.cast(examples_per_gpu * (i + 1), dtype=tf.int32)
                image_a_batch = self.images_a[start:end, ...]
                image_b_batch = self.images_b[start:end, ...]
                ground_truth = self.flows[start:end, ...]

            # Create the loss under this tower.
            with tf.device(gpu):
                # Forward ops.
                final_flow, previous_flows = self.model.get_forward(image_a_batch, image_b_batch,
                                                                    reuse_variables=tf.AUTO_REUSE)
                final_flow_list.append(final_flow)
                accumulate_list_into(previous_flows, previous_flows_list)

                # Loss ops.
                if self.config['fine_tune']:
                    loss, layer_losses = self.model.get_fine_tuning_loss(previous_flows, ground_truth)
                else:
                    loss, layer_losses = self.model.get_training_loss(previous_flows, ground_truth)
                loss_list.append(loss)
                accumulate_list_into(layer_losses, layer_losses_list)

                # Gradient ops.
                grads_and_vars = optimizer.compute_gradients(loss)
                tower_grads_and_vars.append(grads_and_vars)

        # Sets the outputs to be the equivalent of a single-gpu output.
        with tf.name_scope('accumulate_outputs'):
            self.final_flow = tf.concat(final_flow_list, axis=0)
            self.previous_flows = [tf.concat(previous_flows, axis=0) for previous_flows in previous_flows_list]
            self.loss = tf.reduce_mean(tf.stack(loss_list))
            self.layer_losses = [tf.reduce_mean(tf.stack(layer_losses)) for layer_losses in layer_losses_list]

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            averaged_grads_and_vars = average_gradients(tower_grads_and_vars)
            self.train_op = optimizer.apply_gradients(averaged_grads_and_vars, global_step=self.global_step)

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
