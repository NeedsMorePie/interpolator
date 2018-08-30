import tensorflow as tf
from common.utils.flow import get_tf_flow_visualization
from common.utils.multi_gpu import get_available_gpus, TensorIO, create_train_op
from train.pwcnet.trainer import PWCNetTrainer


class PWCNetUnflowTrainer(PWCNetTrainer):
    def __init__(self, model, dataset, session, config, verbose=True):
        self.final_backward_flow = None
        self.previous_backward_flows = None
        self.forward_occlusion_masks = None
        self.backward_occlusion_masks = None
        self.unflow_loss_terms = {}
        super().__init__(model, dataset, session, config, verbose)

    def _create_ops(self):
        """
        Overridden.
        """
        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        unflow_key_id = 'unflow_'

        def build_network_outputs(images_a, images_b, flows):
            # Get the train network.
            bidirectional = self.model.get_bidirectional(images_a, images_b, reuse_variables=tf.AUTO_REUSE)
            final_forward_flow, final_backward_flow, previous_forward_flows, previous_backward_flows = bidirectional

            # Get losses.
            losses = self.model.get_unflow_training_loss(
                images_a, images_b, previous_forward_flows, previous_backward_flows)
            total_loss, layer_losses, forward_occlusion_masks, backward_occlusion_masks, layer_losses_detailed = losses

            output_dict = {
                'loss': TensorIO([total_loss], TensorIO.AVERAGED_SCALAR),
                'layer_losses': TensorIO(layer_losses, TensorIO.AVERAGED_SCALAR),
                'previous_forward_flows': TensorIO(previous_forward_flows, TensorIO.BATCH),
                'previous_backward_flows': TensorIO(previous_backward_flows, TensorIO.BATCH),
                'final_forward_flow': TensorIO([final_forward_flow], TensorIO.BATCH),
                'final_backward_flow': TensorIO([final_backward_flow], TensorIO.BATCH),
                'forward_occlusion_masks': TensorIO(forward_occlusion_masks, TensorIO.BATCH),
                'backward_occlusion_masks': TensorIO(backward_occlusion_masks, TensorIO.BATCH),
            }
            for i, loss_dict in enumerate(layer_losses_detailed):
                for loss_name, value in loss_dict.items():
                    key = unflow_key_id + loss_name + '_layer_' + str(i)
                    assert key not in output_dict
                    output_dict[key] = TensorIO([value], TensorIO.AVERAGED_SCALAR)
            return output_dict

        # Use a helper function to split the batch across multiple GPUs.
        self.train_op, self.global_step, outputs = create_train_op(
            optimizer, build_network_outputs, [self.images_a, self.images_b, self.flows], [],
            available_devices=get_available_gpus(), verbose=True)

        self.final_flow = outputs['final_forward_flow'].first()
        self.final_backward_flow = outputs['final_backward_flow'].first()
        self.previous_flows = outputs['previous_forward_flows'].tensors
        self.previous_backward_flows = outputs['previous_backward_flows'].tensors
        self.loss = outputs['loss'].first()
        self.layer_losses = outputs['layer_losses'].tensors
        self.forward_occlusion_masks = outputs['forward_occlusion_masks'].tensors
        self.backward_occlusion_masks = outputs['backward_occlusion_masks'].tensors

        # Get UnFlow loss terms.
        unflow_loss_keys = [key for key in outputs.keys() if key.startswith(unflow_key_id)]
        for key in unflow_loss_keys:
            self.unflow_loss_terms[key] = outputs[key].first()

    def _make_summaries(self):
        """
        Overridden.
        """
        with tf.name_scope('unflow_summaries'):
            # Backward flows.
            for i, previous_backward_flow in enumerate(self.previous_backward_flows):
                tf.summary.image('backward_flow_' + str(i), get_tf_flow_visualization(previous_backward_flow))
            tf.summary.image('final_backward_flow', get_tf_flow_visualization(self.final_backward_flow))
            # Occlusion masks.
            for i, mask in enumerate(self.forward_occlusion_masks):
                tf.summary.image('forward_occlusion_mask_' + str(i), tf.clip_by_value(mask, 0.0, 1.0))
            for i, mask in enumerate(self.backward_occlusion_masks):
                tf.summary.image('backward_occlusion_mask_' + str(i), tf.clip_by_value(mask, 0.0, 1.0))
            # Detailed loss terms.
            for key, value in self.unflow_loss_terms.items():
                tf.summary.scalar(key, value)

        # Create all other PWC Net summaries and merge them together.
        super()._make_summaries()
