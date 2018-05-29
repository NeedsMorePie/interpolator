import tensorflow as tf


def create_shard_ranges(iter_range, shard_size):
    """
    :param iter_range: List of increasing integers representing a range to iterate over.
    :param shard_size: Maximum shard size.
    :return: List of ranges.
    """
    if shard_size == 0:
        return []

    sharded_iter_ranges = []
    num_shards = int(len(iter_range) / shard_size) + 1
    for shard_id in range(num_shards):
        start_idx = shard_id * shard_size
        end_idx = min((shard_id + 1) * shard_size, len(iter_range))
        if start_idx - end_idx == 0:
            break
        sharded_iter_ranges.append(iter_range[start_idx:end_idx])

    return sharded_iter_ranges


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_bytes_list_feature(bytes_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))

