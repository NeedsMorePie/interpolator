# To resolve some issues with MacOS and matplotlib:
# https://stackoverflow.com/questions/2512225/matplotlib-plots-not-showing-up-in-mac-osx
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import io
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


def print_tensor_shape(x):
    print(x.get_shape().as_list())


def tf_coin_flip(heads_rate):
    rand_val = tf.random_uniform([1], minval=0.0, maxval=1.0)
    is_head = tf.less(rand_val, heads_rate)
    return is_head


# https://github.com/tensorflow/tensorflow/issues/7712
def pelu(x):
    """Parametric Exponential Linear Unit (https://arxiv.org/abs/1605.09332v1)."""
    with tf.variable_scope(x.op.name + '_activation', initializer=tf.constant_initializer(1.0)):
        shape = x.get_shape().as_list()[1:]
        alpha = tf.get_variable('alpha', shape)
        beta = tf.get_variable('beta', shape)
        positive = tf.nn.relu(x) * alpha / (beta + 1e-9)
        negative = alpha * (tf.exp((-tf.nn.relu(-x)) / (beta + 1e-9)) - 1)
        return negative + positive


# https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
def prelu(_x):
    with tf.variable_scope(_x.op.name + '_activation', initializer=tf.constant_initializer(1.0)):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg


def sliding_window_slice(x, slice_locations):
    """
    :param x: The tensor to window slice.
    :param slice_locations: A list. The locations at which we gather values. Values are either 0 or 1.
                            E.g [1, 0, 1] means that at each window offset j, we form [x[j], x[j + 2]].
    :return: The window sliced tensor. Is 1 rank higher than x.
    """
    slice_indices = []
    for i in range(len(slice_locations)):
        if slice_locations[i] == 1:
            slice_indices.append(i)

    sequence_len = len(slice_indices)

    # Get the sliding window indices.
    slice_indices_tensor = tf.constant(slice_indices)
    num_offsets = tf.shape(x)[0] - tf.cast(len(slice_locations) - 1, tf.int32)

    def get_zeros(sequence_len, x):
        return tf.zeros(tf.concat([[1, sequence_len], tf.shape(x)[1:]], axis=0))

    def get_slice(slice_indices_tensor, num_offsets, x):
        num_offsets = tf.maximum(num_offsets, 0)

        # Compute the gather indices for forming sequences.
        tiled = tf.tile(slice_indices_tensor, [num_offsets])
        tiled = tf.expand_dims(tiled, axis=0)
        tiled = tf.reshape(tiled, [num_offsets, -1])
        offsets = tf.expand_dims(tf.range(0, num_offsets), axis=1)
        indices = tiled + offsets
        indices = tf.reshape(indices, [-1])

        # Gather and reshape.
        images = tf.gather(x, indices)
        images = tf.expand_dims(images, axis=0)
        final_shape = tf.concat([[num_offsets, sequence_len], tf.shape(images)[2:]], axis=0)
        images = tf.reshape(images, final_shape)
        return tf.cast(images, tf.float32)

    slices = tf.cond(
        num_offsets > 0,
        true_fn=lambda: get_slice(slice_indices_tensor, num_offsets, x),
        false_fn=lambda: get_zeros(sequence_len, x)
    )
    return slices


# Use this to restore when the variables from the checkpoint and current graph aren't exactly the same.
# Copied from: https://github.com/tensorflow/tensorflow/issues/312
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


# Mostly copied from: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41
class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, graph):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir, graph)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = io.BytesIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def add_run_metadata(self, run_metadata, global_step):
        self.writer.add_run_metadata(run_metadata, 'step%d' % global_step, global_step=global_step)


# Copied from https://github.com/openai/iaf/blob/master/tf_utils/adamax.py.
# Commit 1f09a1b092d7bd406aededc5d5fc64fce766c55e.
class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


def get_available_gpus():
    """
    Gets a list of available GPU device names that can be used in "with tf.device(get_available_gpus()[i]):".
    :return: List of available GPU device names. I.e. ['/device:GPU:0', '/device:GPU:1'].
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Copied from https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/train.py.
# Commit bac9bbaf49be44b9e1c1f004fce4fb04b247763d.
def accumulate_gradients(tower_grads_and_vars):
    """
    Calculate the summed gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    :param tower_grads_and_vars: List of lists of (gradient, variable) tuples. The outer list is over individual
                                 gradients. The inner list is over the gradient calculation for each tower.
    :return: List of pairs of (gradient, variable) where the gradient has been summed across all towers.
    """
    summed_grads = []
    for grad_and_vars in zip(*tower_grads_and_vars):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will sum over below.
                grads.append(expanded_g)
        if len(grads) != 0:
            # Sum over the 'tower' dimension.
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_sum(grad, axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            summed_grads.append(grad_and_var)
    return summed_grads
