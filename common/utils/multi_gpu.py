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


class TensorIO:
    AVERAGED_SCALAR = 0
    SUMMED_SCALAR = 1
    BATCH = 2

    def __init__(self, tensors, tensor_type=AVERAGED_SCALAR):
        """
        :param tensors: 1D list of tensors.
        :param type: Int. Can be either AVERAGED_SCALAR, SUMMED_SCALAR, or BATCH.
                     AVERAGED_SCALAR types are scalars that have been averaged together.
                     SUMMED_SCALAR types are scalars that have been summed together.
                     BATCH types are tensors with the 0th dimension as the batch.
        """
        assert isinstance(tensors, list)
        if len(tensors) > 0:
            assert not isinstance(tensors[0], list)
        self.tensors = tensors
        self.tensor_type = tensor_type

    def first(self):
        """
        :return: The first tensor or None if empty.
        """
        if len(self.tensors) == 0:
            return None
        assert not isinstance(self.tensors[0], list)
        return self.tensors[0]


class AccumulatingTensorIO:
    def __init__(self, tensor_type=TensorIO.AVERAGED_SCALAR):
        """
        :param tensors: 2D array of tensors.
        :param tensor_type: Int. Can be either AVERAGED_SCALAR, SUMMED_SCALAR, or BATCH.
                            AVERAGED_SCALAR types will be averaged into a single scalar.
                            SUMMED_SCALAR types will be summed into a single scalar.
                            BATCH types will be concatenated along the 0th (batch) axis.
        """
        self.tensors = []
        self.tensor_type = tensor_type

    def accumulate(self):
        """
        Accumulates the tensors according to the type.
        :return: TensorIO.
        """
        tensors = []
        if len(self.tensors) > 0:
            assert isinstance(self.tensors[0], list)
            if self.tensor_type == TensorIO.AVERAGED_SCALAR:
                tensors = [tf.reduce_mean(tf.stack(item)) for item in self.tensors]
            elif self.tensor_type == TensorIO.SUMMED_SCALAR:
                tensors = [tf.reduce_sum(tf.stack(item)) for item in self.tensors]
            elif self.tensor_type == TensorIO.BATCH:
                tensors = [tf.concat(item, axis=0) for item in self.tensors]
        return TensorIO(tensors, self.tensor_type)

    def add(self, to_accumulate):
        """
        Adds a list of tensors for accumulation.
        :param to_accumulate: List of tensors.
        :return: Nothing.
        """
        assert isinstance(to_accumulate, list)
        accumulate_list_into(to_accumulate, self.tensors)


def _create_train_op_single_device(optimizer, build_network_outputs, batched_network_args, other_network_args,
                                   available_devices=None, verbose=False):
    """
    See docstring for create_train_op.
    """
    if available_devices is None:
        if verbose:
            print('Creating single device train op for the default device.')
        outputs = build_network_outputs(*(batched_network_args + other_network_args))
    else:
        device_name = available_devices[0]
        if verbose:
            print('Creating single device train op for', device_name)
        with tf.device(device_name):
            outputs = build_network_outputs(*(batched_network_args + other_network_args))
    assert isinstance(outputs, dict)

    with tf.variable_scope('train'):
        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
        train_op = optimizer.minimize(outputs['loss'].first(), global_step=global_step)

    return train_op, global_step, outputs


def create_train_op(optimizer, build_network_outputs, batched_network_args, other_network_args, available_devices=None,
                    verbose=False):
    """
    :param optimizer: Tensorflow optimizer. For example, tf.train.AdamOptimizer(3e-4).
    :param build_network_outputs: A callback to build the network's outputs. It is expected that variable sharing is
                                  enabled under this callback's variable scope.
                                  The first len(batched_network_args) args are batched_network_args and the next
                                  len(other_network_args) are other_network_args. In python speak, this function will be
                                  called like so: build_network_outputs(*(batched_network_args + other_network_args))
                                  This must return a dictionary in the following format:
                                      key: Str<variable name>
                                      value: TensorIO
                                  It is expected that one of the keys is 'loss' and that it corresponds to a scalar
                                  tensor. Thus, the minimum output is {'loss': TensorIO([loss_tensor])}.
    :param batched_network_args: List of Tensors. These inputs will be sliced into sub-batches to be fed into
                                 different GPU devices. It is assumed that dimension 0 is the batch dimension and that
                                 all Tensors here have the same batch size.
    :param other_network_args: List. These can be anything. Each device will receive the same copy of these args.
    :param available_devices: List of str. I.e. ['/device:GPU:0', '/device:GPU:1']. If there is only 1 device in the
                              list, then the function simply returns build_network_outputs(). If this is None, that is
                              the equivalent of using the single default device.
    :param verbose: Bool. Whether to make print statements.
    :return: train_op: Tensorflow op.
             global_step: Tensor (variable). Int variable incremented every time train_op is run.
             output_dict: Dict. This can be treated exactly the same as the return value of build_network_outputs() for
                          one GPU. It is in the following format:
                             key: Str<variable name>
                             value: TensorIO
    """
    # Do single-device version of this function and bypass all the complex accumulating.
    if available_devices is None or len(available_devices) == 1:
        return _create_train_op_single_device(optimizer, build_network_outputs, batched_network_args,
                                              other_network_args, available_devices, verbose)

    # Get the number of examples per GPU.
    with tf.name_scope('examples_per_gpu'):
        examples_per_gpu = None
        if len(batched_network_args) > 0:
            batch_size = tf.shape(batched_network_args[0])[0]
            num_gpus = len(available_devices)
            examples_per_gpu = tf.cast(batch_size / num_gpus, dtype=tf.int32)

    # Accumulation variables.
    accumulation_dict = {}
    tower_grads_and_vars = []

    # Create the towers.
    for i, device in enumerate(available_devices):
        if verbose:
            print('Creating tower for', device)

        with tf.name_scope('batch_distribution'):
            sub_batched_network_args = []
            if examples_per_gpu is not None:
                start = tf.cast(examples_per_gpu * i, dtype=tf.int32)
                end = tf.cast(examples_per_gpu * (i + 1), dtype=tf.int32)
                sub_batched_network_args = [arg[start:end, ...] for arg in batched_network_args]

        # Create the loss under this tower.
        with tf.device(device):
            outputs = build_network_outputs(*(sub_batched_network_args + other_network_args))

            # Accumulate outputs.
            for key, value in outputs.items():
                tensor_list = value.tensors
                accumulation_type = value.tensor_type
                if key not in accumulation_dict:
                    accumulation_dict[key] = AccumulatingTensorIO(accumulation_type)
                assert accumulation_dict[key].tensor_type == accumulation_type
                accumulation_dict[key].add(tensor_list)

            # Create the gradient ops.
            grads_and_vars = optimizer.compute_gradients(outputs['loss'].first())
            tower_grads_and_vars.append(grads_and_vars)

    # Create the train op.
    with tf.variable_scope('train'):
        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
        averaged_grads_and_vars = average_gradients(tower_grads_and_vars)
        train_op = optimizer.apply_gradients(averaged_grads_and_vars, global_step=global_step)

    # Sets the outputs to be the equivalent of a single-gpu output.
    with tf.name_scope('accumulate_outputs'):
        output_dict = {}
        for key, value in accumulation_dict.items():
            output_dict[key] = value.accumulate()

    return train_op, global_step, output_dict
