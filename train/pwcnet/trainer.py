import tensorflow as tf
from data.flow.flowdata import FlowDataSet
from pwcnet.model import PWCNet
from train.trainer import Trainer


# TODO: Checkpoint saving.
# TODO: Tensorboard.
class PWCNetTrainer(Trainer):
    def __init__(self, model, dataset, session, config, verbose=True):
        super().__init__(model, dataset, session, config, verbose)
        assert isinstance(model, PWCNet)
        assert isinstance(dataset, FlowDataSet)

        dataset.load()
        train_images_a, train_images_b, train_flows = dataset.get_next_train_batch()
        valid_images_a, valid_images_b, valid_flows = dataset.get_next_validation_batch()

        # Get the train network.
        train_final_flow, train_previous_flows = model.get_forward(train_images_a, train_images_b,
                                                                   reuse_variables=tf.AUTO_REUSE)
        self.train_loss, self.train_layer_losses = model.get_training_loss(train_previous_flows, train_flows)

        # Get the validatio network.
        valid_final_flow, valid_previous_flows = model.get_forward(valid_images_a, valid_images_b,
                                                                   reuse_variables=tf.AUTO_REUSE)
        self.valid_loss, self.valid_layer_losses = model.get_training_loss(valid_previous_flows, valid_flows)

        # Get the optimizer.
        self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
        self.train_op = tf.train.AdamOptimizer(config['learning_rate']).minimize(
            self.train_loss, global_step=self.global_step)

    def train_for(self, iterations):
        """
        Overridden.
        """
        avg_loss = 0.0
        for i in range(iterations):
            loss, _ = self.session.run([self.train_loss, self.train_op])
            avg_loss += loss
        avg_loss /= float(iterations)

    def validate(self, iterations):
        """
        Overridden.
        """
        avg_loss = 0.0
        for i in range(iterations):
            loss = self.session.run(self.valid_loss)
            avg_loss += loss
        avg_loss /= float(iterations)
