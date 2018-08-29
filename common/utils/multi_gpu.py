import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    """
    Gets a list of available GPU device names that can be used in "with tf.device(get_available_gpus()[i]):".
    :return: List of available GPU device names. I.e. ['/device:GPU:0', '/device:GPU:1'].
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Copied from https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/train.py.
# Commit bac9bbaf49be44b9e1c1f004fce4fb04b247763d.
def average_gradients(tower_grads_and_vars):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    :param tower_grads_and_vars: List of lists of (gradient, variable) tuples. The outer list is over individual
                                 gradients. The inner list is over the gradient calculation for each tower.
    :return: List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
    """
    averaged_grads = []
    for grad_and_vars in zip(*tower_grads_and_vars):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        if len(grads) != 0:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            averaged_grads.append(grad_and_var)
    return averaged_grads


def accumulate_list_into(items, items_list):
    """
    The primary use case of this function is to accumulate multiple lists of tensors into a list of a list of tensors
    for joining together the same outputs from multiple GPUs.
    This is best explained by an example:
    items_list = []
    items = [1, 2]
    accumulate_list_into(items, items_list)  # items_list is now [[1], [2]].
    items = [3, 4]
    accumulate_list_into(items, items_list)  # items_list is now [[1, 3], [2, 4]].
    :param items: List of items.
    :param items_list: Empty list.
    :return: Nothing.
    """
    for j, item in enumerate(items):
        items_list.append([item]) if len(items_list) < len(items) else items_list[j].append(item)
