import tensorflow as tf


# Dataset interface.
class DataSet:
    def __init__(self, directory, batch_size, validation_size, training_augmentations=True):
        self.directory = directory
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.training_augmentations = training_augmentations
        self.verbose = False

    def set_verbose(self, verbose):
        """
        :param verbose: Whether to have print statements everywhere.
        :return: Nothing.
        """
        self.verbose = verbose

    def get_train_file_names(self):
        """
        :return: List of string.
        """
        raise NotImplementedError('get_train_file_names() is not implemented.')

    def get_validation_file_names(self):
        """
        :return: List of string.
        """
        raise NotImplementedError('get_validation_file_names() is not implemented.')

    def preprocess_raw(self, shard_size):
        """
        Takes a raw dataset (i.e. directories of images) and processes them into an intermediate format (i.e.
        TFRecords or numpy arrays).
        Output file is named get_train_file_names() and get_validation_file_names().
        :param shard_size: Number of elements in each shard.
        :return: Nothing.
        """
        raise NotImplementedError('preprocess_raw() is not implemented.')

    def load(self, session):
        """
        Loads the intermediate format.
        :param session: Tensorflow session.
        :return: Nothing.
        """
        assert isinstance(session, tf.Session)
        raise NotImplementedError('load() is not implemented.')

    def get_next_batch(self):
        """
        Gets the tensors for the next batch.
        :return: Tensors.
        """
        raise NotImplementedError('get_next_batch() is not implemented.')

    def get_train_feed_dict(self):
        """
        Gets the feed_dict for the training dataset.
        Usage: Getting the next batch of data:
            sess.run(dataset.get_next_batch(), feed_dict=dataset.get_train_feed_dict()).
        :return: Dictionary.
        """
        raise NotImplementedError('get_train_feed_dict() is not implemented.')

    def get_validation_feed_dict(self):
        """
        Gets the feed_dict for the validation dataset.
        Usage: Getting the next batch of data:
            dataset.init_validation_data(sess)  # Run once per validation epoch.
            sess.run(dataset.get_next_batch(), feed_dict=dataset.get_validation_feed_dict()).
        :return: Dictionary.
        """
        raise NotImplementedError('get_validation_feed_dict() is not implemented.')

    def init_validation_data(self, session):
        """
        :param session: Tensorflow session.
        :return: Nothing.
        """
        assert isinstance(session, tf.Session)
        raise NotImplementedError('init_validation_data() is not implemented.')
