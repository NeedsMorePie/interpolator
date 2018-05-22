import unittest
import os
from utils.flow import read_flow_file
from utils.img import read_image, show_image
from forward_warp.warp import forward_warp
import numpy as np

VISUALIZE = False


class TestForwardWarp(unittest.TestCase):
    def test_visualization(self):
        if not VISUALIZE:
            return

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(cur_dir, '../')
        flow_ab = read_flow_file(root_dir + 'pwcnet/warp/test_data/flow_ab.flo')
        img_a = read_image(root_dir + 'pwcnet/warp/test_data/image_a.png', as_float=True)
        warped = forward_warp(img_a, flow_ab, 1.0)
        show_image(warped)
