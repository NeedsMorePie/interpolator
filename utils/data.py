import tensorflow as tf


def create_shard_ranges(iter_range, shard_size):
    """
    :param iter_range: List of increasing integers representing a range to iterate over.
    :param shard_size: Maximum shard size.
    :return: List of ranges.
    """
    sharded_iter_ranges = []
    current_shard_ids = []
    first_example = True
    for i in iter_range:
        if i % shard_size == 0 and not first_example:
            # New shard.
            sharded_iter_ranges.append(current_shard_ids)
            current_shard_ids = []
        first_example = False
        current_shard_ids.append(i)
    # Append the last shard if there is one.
    if len(current_shard_ids) != 0:
        sharded_iter_ranges.append(current_shard_ids)
    return sharded_iter_ranges


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
