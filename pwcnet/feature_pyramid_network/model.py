import tensorflow as tf
from common.models import ConvNetwork


class FeaturePyramidNetwork(ConvNetwork):
    def __init__(self, name='feature_pyramid_network', layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 regularizer=None, dense_net=False):
        """
        :param name: Str. For variable scoping.
        :param layer_specs: See parent class.
        :param activation_fn: Tensorflow activation function.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        :param dense_net: Bool.
        """
        super().__init__(layer_specs=layer_specs,
                         activation_fn=activation_fn, regularizer=regularizer, padding='SAME', dense_net=dense_net)

        self.name = name
        if layer_specs is None:
            # PWC-Net default.
            self.layer_specs = [[3, 16, 1, 2],
                                [3, 16, 1, 1],
                                [3, 32, 1, 2],
                                [3, 32, 1, 1],
                                [3, 64, 1, 2],
                                [3, 64, 1, 1],
                                [3, 96, 1, 2],
                                [3, 96, 1, 1],
                                [3, 128, 1, 2],
                                [3, 128, 1, 1],
                                [3, 192, 1, 2],
                                [3, 192, 1, 1]]
        else:
            self.layer_specs = layer_specs

    def get_forward(self, image, reuse_variables=False):
        """
           input
             |
         [LAYER 0]
            ...
        [LAYER N-1]
             |
        final_features
        :param image: Tensor. Shape [batch_size, H, W, 3].
        :param reuse_variables: Bool. Whether to reuse the variables.
        :return: final_features: features of shape [batch_size, H, W, 192].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            return self._get_conv_tower(image)
