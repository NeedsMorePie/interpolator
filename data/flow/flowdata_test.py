import numpy as np
import os
import os.path
import tensorflow as tf
import unittest
from data.flow.flowdata import FlowDataSet


class TestFlowDataSet(unittest.TestCase):
    def setUp(self):
        self.data_directory = os.path.join('data', 'flow', 'test_data')
        self.flow_directory = os.path.join(self.data_directory, 'test_flows')
        self.image_directory = os.path.join(self.data_directory, 'test_images')
        # No data augmentation so that the tests are deterministic.
        self.data_set = FlowDataSet(self.data_directory, batch_size=2, validation_size=1, training_augmentations=False)

        # Test paths.
        self.expected_image_a_paths = [os.path.join(self.image_directory, 'set_a', 'image_0000.png'),
                                       os.path.join(self.image_directory, 'set_a', 'image_0001.png'),
                                       os.path.join(self.image_directory, 'set_a', 'image_0002.png'),
                                       os.path.join(self.image_directory, 'set_a', 'image_0003.png'),
                                       os.path.join(self.image_directory, 'set_b', 'image_0001.png')]
        self.expected_image_b_paths = [os.path.join(self.image_directory, 'set_a', 'image_0001.png'),
                                       os.path.join(self.image_directory, 'set_a', 'image_0002.png'),
                                       os.path.join(self.image_directory, 'set_a', 'image_0003.png'),
                                       os.path.join(self.image_directory, 'set_a', 'image_0004.png'),
                                       os.path.join(self.image_directory, 'set_b', 'image_0002.png')]
        self.expected_flow_paths = [os.path.join(self.flow_directory, 'set_a', 'flow_0000.flo'),
                                    os.path.join(self.flow_directory, 'set_a', 'flow_0001.flo'),
                                    os.path.join(self.flow_directory, 'set_a', 'flow_0002.flo'),
                                    os.path.join(self.flow_directory, 'set_a', 'flow_0003.flo'),
                                    os.path.join(self.flow_directory, 'set_b', 'flow_0001.flo')]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def test_data_paths(self):
        """
        Test that the data paths make sense.
        """
        image_a_paths, image_b_paths, flow_paths = self.data_set._get_data_paths()
        self.assertListEqual(image_a_paths, self.expected_image_a_paths)
        self.assertListEqual(image_b_paths, self.expected_image_b_paths)

    def test_data_read_write(self):
        self.data_set.preprocess_raw(shard_size=2)
        output_paths = self.data_set.get_train_file_names() + self.data_set.get_validation_file_names()
        [self.assertTrue(os.path.isfile(output_path)) for output_path in output_paths]
        # The train set should have been sharded, so there should be 3 files.
        self.assertEquals(len(output_paths), 3)

        self.data_set.load(self.sess)
        next_images_a, next_images_b, next_flows = self.data_set.get_next_batch()
        images_1_a, images_1_b, flows = self.sess.run([next_images_a, next_images_b, next_flows],
                                                      feed_dict=self.data_set.get_train_feed_dict())
        self.assertTupleEqual(images_1_a.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(images_1_b.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))
        images_2_a, images_2_b, flows = self.sess.run([next_images_a, next_images_b, next_flows],
                                                      feed_dict=self.data_set.get_train_feed_dict())
        self.assertTupleEqual(images_2_a.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(images_2_b.shape, (2, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (2, 436, 1024, 2))

        # Make sure all the images are different (i.e. read correctly).
        self.assertFalse(np.allclose(images_1_a[0], images_1_a[1]))
        self.assertFalse(np.allclose(images_2_a[0], images_1_a[1]))
        self.assertFalse(np.allclose(images_1_a[0], images_2_a[1]))
        self.assertFalse(np.allclose(images_2_a[0], images_2_a[1]))
        # self.assertTrue(np.max(images_1_a[0]) <= 1.0)

        # Validation data size is 1, so even though the dataset batch size is 2, the validation batch size is 1.
        self.data_set.init_validation_data(self.sess)
        images_a, images_b, flows = self.sess.run([next_images_a, next_images_b, next_flows],
                                                  feed_dict=self.data_set.get_validation_feed_dict())
        self.assertTupleEqual(images_a.shape, (1, 436, 1024, 3))
        self.assertTupleEqual(images_b.shape, (1, 436, 1024, 3))
        self.assertTupleEqual(flows.shape, (1, 436, 1024, 2))
        # If init_validation_data isn't called again, validation dataset has hit the end of its epoch and will be out
        # of range.
        end_of_dataset = False
        try:
            self.sess.run([next_images_a, next_images_b, next_flows],
                          feed_dict=self.data_set.get_validation_feed_dict())
        except tf.errors.OutOfRangeError:
            end_of_dataset = True
        self.assertTrue(end_of_dataset)

    def tearDown(self):
        output_paths = self.data_set.get_train_file_names() + self.data_set.get_validation_file_names()
        for output_path in output_paths:
            if os.path.isfile(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
