import numpy as np
import cv2
from utils.flow import read_flow_file, show_flow_image
from utils.img import read_image, show_image


def forward_warp(image, flow, t):
    """
    See section 3 in https://arxiv.org/pdf/1711.05890.pdf
    :param image: Image to be warped, of shape [H, W, channels].
    :param flow: Un-normalized flow (in image pixel units), of shape [H, W, 2].
    :param t: Float that specifies interpolation degree. 0 for not flowed, 1 for flow at full optical flow length.
    """
    height, width, channels = image.shape
    warped = np.zeros(image.shape)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                v = t * flow[y][x]
                l = np.array([v[1], v[0]]) + np.array([y, x])

                # Inverse bi-linear interpolation.
                top_left = [np.floor(l[0]), np.floor(l[1])]
                top_right = [np.floor(l[0]), np.ceil(l[1])]
                bot_right = [np.ceil(l[0]), np.ceil(l[1])]
                bot_left = [np.ceil(l[0]), np.floor(l[1])]
                locations = [top_left, top_right, bot_right, bot_left]
                for location in locations:
                    if location[0] < 0 or location[0] >= height or location[1] < 0 or location[1] >= width:
                        continue

                    weight = (1.0 - np.abs(l[0] - location[0])) * (1.0 - np.abs(l[1] - location[1]))
                    location = np.array(location).astype(np.int32)
                    warped[location[0]][location[1]][c] += weight * image[y][x][c]

    return warped