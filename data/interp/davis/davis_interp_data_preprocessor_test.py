import os
import os.path
import tensorflow as tf
import unittest
from data.interp.davis.davis_interp_data_preprocessor import DavisDataSetPreprocessor


class TestDavisDataSet(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(__file__)
        self.data_directory = os.path.join(cur_dir, 'test_data')

        # Test paths.
        self.expected_image_paths_0 = [
            os.path.join(self.data_directory, 'breakdance-flare', '00000.jpg'),
            os.path.join(self.data_directory, 'breakdance-flare', '00001.jpg'),
            os.path.join(self.data_directory, 'breakdance-flare', '00002.jpg'),
            os.path.join(self.data_directory, 'breakdance-flare', '00003.jpg'),
            os.path.join(self.data_directory, 'breakdance-flare', '00004.jpg'),
            os.path.join(self.data_directory, 'breakdance-flare', '00005.jpg'),
            os.path.join(self.data_directory, 'breakdance-flare', '00006.jpg')
        ]

        self.expected_image_paths_1 = [
            os.path.join(self.data_directory, 'dog-agility', '00000.jpg'),
            os.path.join(self.data_directory, 'dog-agility', '00001.jpg'),
            os.path.join(self.data_directory, 'dog-agility', '00002.jpg'),
        ]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_data_paths(self):
        """
        Test that the data paths make sense.
        """
        data_set = DavisDataSetPreprocessor(self.data_directory, [[1]])
        image_paths = data_set.get_data_paths(self.data_directory)
        self.assertListEqual(image_paths[0], self.expected_image_paths_0)
        self.assertListEqual(image_paths[1], self.expected_image_paths_1)


if __name__ == '__main__':
    unittest.main()
