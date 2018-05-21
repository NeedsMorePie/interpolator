# Dataset interface.
class DataSet:
    def __init__(self, directory, batch_size, validation_size):
        self.directory = directory
        self.batch_size = batch_size
        self.validation_size = validation_size
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

    def load(self):
        """
        Loads the intermediate format.
        :return: Nothing.
        """
        raise NotImplementedError('load() is not implemented.')

    def get_next_train_batch(self):
        """
        Gets the next batch as either tensors or numpy arrays.
        :return: Variables for the next batch.
        """
        raise NotImplementedError('get_next_batch() is not implemented.')

    def get_next_validation_batch(self):
        """
        Gets the next batch as either tensors or numpy arrays.
        :return: Variables for the next batch.
        """
        raise NotImplementedError('get_next_batch() is not implemented.')
