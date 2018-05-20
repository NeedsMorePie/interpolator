# Dataset interface.
class DataSet:
    def __init__(self, directory, batch_size, validation_size):
        self.directory = directory
        self.batch_size = batch_size
        self.validation_size = validation_size

    def get_processed_file_names(self):
        """
        preprocess_raw() uses this as the filename.
        :return: List of string.
        """
        raise NotImplementedError('get_processed_file_name() is not implemented.')

    def preprocess_raw(self):
        """
        Takes a raw dataset (i.e. directories of images) and processes them into an intermediate format (i.e.
        TFRecords or numpy arrays).
        Output file is named get_processed_file_name().
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
