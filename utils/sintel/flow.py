import numpy as np


SINTEL_FLOW_CHANNELS = 2
SINTEL_TAG_FLOAT = 202021.25


def read_flow_file(file_name):
    """
    :param file_name: str.
    :return: Numpy array of shape (height, width, SINTEL_FLOW_CHANNELS).
        Returns None if the tag in the file header was invalid.
    """
    with open(file_name, 'rb') as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if tag != SINTEL_TAG_FLOAT:
            return None
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]

        num_image_floats = width * height * SINTEL_FLOW_CHANNELS
        image = np.fromfile(f, dtype=np.float32, count=num_image_floats)
        image = image.reshape((height, width, SINTEL_FLOW_CHANNELS))

        return image
