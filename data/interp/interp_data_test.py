import numpy as np
import os
import os.path
import tensorflow as tf
import unittest
from data.interp.interp_data import InterpDataSet


class TestInterpDataSet(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(__file__)
        self.data_directory = os.path.join(cur_dir, 'test_data')
        self.data_set = InterpDataSet(self.data_directory, batch_size=2)

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
        image_paths = self.data_set._get_data_paths()
        self.assertListEqual(image_paths[0], self.expected_image_paths_0)
        self.assertListEqual(image_paths[1], self.expected_image_paths_1)

    def test_data_read_write(self):
        self.data_set.preprocess_raw(shard_size=1)

        output_paths = self.data_set.get_train_file_names() + self.data_set.get_validation_file_names()
        [self.assertTrue(os.path.isfile(output_path)) for output_path in output_paths]

        # For a shard size of 1 the number of files is the number of video shots (or the number of sub-folders).
        self.assertEquals(len(output_paths), 2)

        #
        # self.data_set.load(self.sess)
        # next_images_a, next_images_b, next_flows = self.data_set.get_next_batch()
        # images_1_a, images_1_b, flows = self.sess.run([next_images_a, next_images_b, next_flows],
        #                                               feed_dict=self.data_set.get_train_feed_dict())
        # self.assertTupleEqual(images_1_a.shape, (2, 436, 1024, 3))
        # self.assertTupleEqual(images_1_b.shape, (2, 436, 1024, 3))
        # self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))
        # images_2_a, images_2_b, flows = self.sess.run([next_images_a, next_images_b, next_flows],
        #                                               feed_dict=self.data_set.get_train_feed_dict())
        # self.assertTupleEqual(images_2_a.shape, (2, 436, 1024, 3))
        # self.assertTupleEqual(images_2_b.shape, (2, 436, 1024, 3))
        # self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))
        #
        # # Make sure all the images are different (i.e. read correctly).
        # self.assertFalse(np.allclose(images_1_a[0], images_1_a[1]))
        # self.assertFalse(np.allclose(images_2_a[0], images_1_a[1]))
        # self.assertFalse(np.allclose(images_1_a[0], images_2_a[1]))
        # self.assertFalse(np.allclose(images_2_a[0], images_2_a[1]))
        # self.assertTrue(np.max(images_1_a[0]) <= 1.0)
        #
        # # Validation data size is 1, so even though the dataset batch size is 2, the validation batch size is 1.
        # self.data_set.init_validation_data(self.sess)
        # images_a, images_b, flows = self.sess.run([next_images_a, next_images_b, next_flows],
        #                                           feed_dict=self.data_set.get_validation_feed_dict())
        # self.assertTupleEqual(images_a.shape, (1, 436, 1024, 3))
        # self.assertTupleEqual(images_b.shape, (1, 436, 1024, 3))
        # self.assertTupleEqual(flows.shape, (1, 436, 1024, 2))
        # # If init_validation_data isn't called again, validation dataset has hit the end of its epoch and will be out
        # # of range.
        # end_of_dataset = False
        # try:
        #     self.sess.run([next_images_a, next_images_b, next_flows],
        #                   feed_dict=self.data_set.get_validation_feed_dict())
        # except tf.errors.OutOfRangeError:
        #     end_of_dataset = True
        # self.assertTrue(end_of_dataset)

    # def tearDown(self):
        # output_paths = self.data_set.get_train_file_names() + self.data_set.get_validation_file_names()
        # for output_path in output_paths:
        #     if os.path.isfile(output_path):
        #         os.remove(output_path)

    #def test_sliding_window_slice(self):
    #    x =

if __name__ == '__main__':
    unittest.main()
