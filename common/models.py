import numpy as np
import tensorflow as tf


_default = object()


# Allows saving and restoring all network parameters its scope into an npz file.
# Provides a way to serialize the network in a minimal form without any tensorflow metadata.
# If the network is fully convolutional, this also allows restoring weights into a network with larger inputs/outputs.
class RestorableNetwork():
    def __init__(self, name):
        """
        :param name: Str. For variable scoping.
        """
        self.name = name

        # Dictionary with key = variable name and value = (assign_op, placeholder).
        self._assign_ops = {}

    def save_to(self, file_path, sess):
        """
        Serializes the network into an npz file.
        :param file_path: Str.
        :param sess: Tensorflow session.
        :return: Nothing.
        """
        save_dict = self.get_save_np(sess)
        np.savez(file_path, **save_dict)

    def get_save_np(self, sess):
        """
        Creates a dict of np arrays corresponding to the weights of each variable.
        :param sess: Tensorflow session.
        :return: Dict of np arrays. Key is the name of the variable.
        """
        trainable_vars = tf.trainable_variables(self.name)
        trainable_vars_np = sess.run(trainable_vars)
        # Create a dict of variable_name:variable_np.
        save_dict = {}
        for i, var in enumerate(trainable_vars):
            var_name = var.name
            save_dict[var_name] = trainable_vars_np[i]
        return save_dict

    def restore_from(self, file_path, sess, scope_prefix=''):
        """
        Deserializes a network from an npz file.
        Instructions on how to handle changes to network name and scope:
            If the network's name has changed:
                Call restore_from_np(rename_np_dict(np.load(file_path), <old_name>, <new_name>), sess)
            If the network's scope has gotten deeper:
                Call restore_from(file_path, sess, scope_prefix=<new_scope>)
            If the network's name and scope has changed:
                Call restore_from_np(rename_np_dict(np.load(file_path), <old_name>, <new_name>), sess,
                                     scope_prefix=<new_scope>)
         Note that we only "officially" support the scope getting deeper, because it doesn't make sense for a trained
         and saved network to become shallower -- networks shouldn't be trained in a scope deeper than their name.
        :param file_path: Str.
        :param sess: Tensorflow session.
        :param scope_prefix: Adds a prefix to the scope filter when getting the vars to restore. If it is '', then no
            prefix will be used. The npz file does not need to be modified if name of the network is still the same.
        :return: Nothing.
        """
        var_dict = np.load(file_path)
        self.restore_from_np(var_dict, sess, scope_prefix)

    def restore_from_np(self, var_dict, sess, scope_prefix=''):
        """
        Restores the network from the var_dict.
        :param var_dict: Dict of np arrays. Key is the name of the variable.
        :param sess: Tensorflow session.
        :param scope_prefix: Adds a prefix to the scope filter when getting the vars to restore. If it is '', then no
            prefix will be used. The var_dict does not need to be modified if name of the network is still the same.
        :return: Nothing
        """
        def _append_prefix(name):
            return scope_prefix + '/' + name
        scope_filter = self.name
        if scope_prefix != '':
            scope_filter = _append_prefix(self.name)
            var_dict = {_append_prefix(key): value for (key, value) in var_dict.items()}
        with tf.name_scope(self.name + '_assign_ops'):
            trainable_vars = tf.trainable_variables(scope_filter)
            assert len(trainable_vars) == len(var_dict.keys())
            feed_dict = {}
            assign_ops = []
            for var in trainable_vars:
                var_name = var.name
                assert var_name in var_dict
                var_np = var_dict[var_name]
                assign_op, placeholder = self.get_assign_op(var)
                assign_ops.append(assign_op)
                assert placeholder not in feed_dict
                feed_dict[placeholder] = var_np
            sess.run(assign_ops, feed_dict=feed_dict)

    def get_assign_op(self, var):
        """
        :param var: Tensorflow variable.
        :return: Operation, placeholder.
        """
        var_name = var.name
        if var_name not in self._assign_ops:
            ph = tf.placeholder(dtype=tf.float32)
            op = tf.assign(var, ph, validate_shape=True)
            self._assign_ops['var_name'] = op, ph
        return self._assign_ops['var_name']

    @staticmethod
    def rename_np_dict(var_dict, old_network_name, new_network_name):
        """
        Renames the np dict so a new network with a different name but same architecture can restore from it.
        :param old_network_name: Str. Name of the old network.
        :param new_network_name: Str. Name of the new network.
        :return: New var dict.
        """
        new_dict = {}
        for key, value in var_dict.items():
            new_key = key.replace(old_network_name, new_network_name)
            new_dict[new_key] = value
        return new_dict


class ConvNetwork(RestorableNetwork):
    def __init__(self, name, layer_specs=None,
                 activation_fn=tf.nn.leaky_relu,
                 last_activation_fn=_default,
                 regularizer=None, padding='SAME', dense_net=False):
        """
        Generic conv-net
        :param name: Str. For variable scoping.
        :param layer_specs: Array of shape [num_layers, 4].
                            The second dimension consists of [kernel_size, num_output_features, dilation, stride].
        :param activation_fn: Tensorflow activation function.
        :param last_activation_fn: Tensorflow activation function. Applied after the final convolution of activation_fn,
                                   in place of activation_fn. Defaults to the value of activation_fn.
        :param regularizer: Tf regularizer such as tf.contrib.layers.l2_regularizer.
        :param padding: Str. Either 'SAME' or 'VALID' case insensitive.
        :param dense_net: Bool. If true, then it is expected that all layers have the same width and height.
        """
        super().__init__(name)
        self.layer_specs = layer_specs
        self.activation_fn = activation_fn
        self.regularizer = regularizer
        self.padding = padding
        self.dense_net = dense_net

        if last_activation_fn == _default:
            self.last_activation_fn = self.activation_fn
        else:
            self.last_activation_fn = last_activation_fn

    def _get_conv_tower(self, features):
        """
        :param features: Tensor. Feature map of shape [batch_size, H, W, num_features].
        :return: final_output: tensor of shape [batch_size, H, W, num_output_features].
                 layer_outputs: array of layer intermediate conv outputs. Length is len(layer_specs) + 1.
        """
        layer_outputs = []

        # Create the network layers.
        previous_output = features
        for i, layer_spec in enumerate(self.layer_specs):
            # Get specs.
            kernel_size = layer_spec[0]
            num_output_features = layer_spec[1]
            dilation = layer_spec[2]
            stride = layer_spec[3]

            is_last_layer = i == len(self.layer_specs) - 1
            activation_fn = self.last_activation_fn if is_last_layer else self.activation_fn
            if self.dense_net and i != 0:
                # Dense-net layer input consists of all previous layer outputs.
                assert previous_output == layer_outputs[-1]
                assert features not in layer_outputs
                inputs = tf.concat(layer_outputs + [features], axis=-1)
            else:
                inputs = previous_output

            # Create the convolution layer.
            previous_output = tf.layers.conv2d(inputs=inputs,
                                               filters=num_output_features,
                                               kernel_size=[kernel_size, kernel_size],
                                               strides=(stride, stride),
                                               padding='SAME',
                                               dilation_rate=(dilation, dilation),
                                               activation=None,
                                               kernel_regularizer=self.regularizer,
                                               bias_regularizer=self.regularizer,
                                               name='conv_' + str(i))

            if activation_fn is not None:
                previous_output = activation_fn(previous_output)

            layer_outputs.append(previous_output)

        final_output = previous_output
        return final_output, layer_outputs

    def get_forward_conv(self, features, reuse_variables=tf.AUTO_REUSE):
        """
        Public API for getting the forward ops.
        :param features: Feature map or images.
        :param reuse_variables: Whether to reuse the variables under the scope.
        :return: final_output, layer_outputs.
        """
        with tf.variable_scope(self.name, reuse=reuse_variables):
            return self._get_conv_tower(features)

