import cv2
import matplotlib.pyplot as plt
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


def show_flow_image(flow_img):
    """
    Opens a window displaying an optical flow image.
    Uses the HSV color wheel for the visualization.
    It appears that red means movement to the right, blue means movement to the left, yellow means movement downward,
    purple means movement upward.
    :param flow_img: Numpy array of shape (Height, Width, 2).
    :return: Nothing.
    """
    abs_image = np.abs(flow_img)
    flow_mean = np.mean(abs_image)
    flow_std = np.std(abs_image)

    # Apply some kind of normalization. Divide by the perceived maximum (mean + std)
    flow_img = flow_img / (flow_mean + flow_std + 1e-10)

    # Get the angle and radius.
    # sqrt(x^2 + y^2).
    radius = np.sqrt(np.square(flow_img[..., 0]) + np.square(flow_img[..., 1]))
    radius_clipped = np.clip(radius, 0.0, 1)
    # Arctan2(-y, -x) / pi so that it's between [-1, 1].
    angle = np.arctan2(-flow_img[..., 1], -flow_img[..., 0]) / np.pi

    # Hue is between [0, 1] but scaled by (179.0 / 255.0) because hue is usually between [0, 179].
    hue = np.clip((angle + 1.0) / 2.0, 0.0, 1.0) * (179.0 / 255.0)
    # Saturation is always 0.75 (it's more aesthetic).
    saturation = np.ones(shape=hue.shape, dtype=np.float) * 0.75
    # Value (brightness), is the radius (magnitude) of the flow.
    value = radius_clipped

    hsv_img = (np.clip(np.stack([hue, saturation, value], axis=-1) * 255, 0, 255)).astype(dtype=np.uint8)
    rgb_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    plt.imshow(rgb_image)
    plt.show()
