import unittest
import os
from utils.flow import read_flow_file
from utils.img import read_image, show_image
from experiments.forward_warp import forward_warp
import matplotlib.image as mpimg
import numpy as np
import cv2

VISUALIZE = True


class TestForwardWarp(unittest.TestCase):
    def test_visualization(self):
        if not VISUALIZE:
            return

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(cur_dir, '../')
        flow_ab = read_flow_file(root_dir + 'pwcnet/warp/test_data/flow_ab.flo')
        img_a = read_image(root_dir + 'pwcnet/warp/test_data/image_a.png', as_float=True)
        warped = forward_warp(img_a, flow_ab, 1.0)
        warped = np.clip(warped, 0.0, 1.0)
        show_image(warped)

        # For writing to video.
        # height = img_a.shape[0]
        # width = img_a.shape[1]
        # writer = cv2.VideoWriter(cur_dir + '/outputs/video.avi', cv2.VideoWriter_fourcc(*"MJPG"), 20, (width, height))
        #
        # steps = 20
        # for i in range(steps):
        #     t = i * (1.0 / float(steps))
        #     warped = forward_warp(img_a, flow_ab, t)
        #     warped = np.clip(warped, 0.0, 1.0)
        #     output_path = cur_dir + "/outputs/out-%.2f.png" % t
        #     mpimg.imsave(output_path, warped)
        #     writer.write(cv2.imread(output_path))
        #
        # writer.release()
