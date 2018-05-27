import numpy as np
import os
import os.path
import tensorflow as tf
import unittest
from utils.img import show_image
from data.interp.interp_data import InterpDataSet

VISUALIZE = False

class TestInterpDataSet(unittest.TestCase):
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
        data_set = InterpDataSet(self.data_directory, [[1]], batch_size=2)
        image_paths = data_set._get_data_paths()
        self.assertListEqual(image_paths[0], self.expected_image_paths_0)
        self.assertListEqual(image_paths[1], self.expected_image_paths_1)

    def test_data_read_write(self):
        data_set = InterpDataSet(self.data_directory, [[1]], batch_size=2)
        data_set.preprocess_raw(shard_size=1)

        output_paths = data_set.get_tf_record_names()
        [self.assertTrue(os.path.isfile(output_path)) for output_path in output_paths]

        # For a shard size of 1 the number of files is the number of video shots (or the number of sub-folders).
        self.assertEqual(len(output_paths), 2)

        data_set.load(self.sess)

        next_sequence_tensor, next_sequence_timing_tensor = data_set.get_next_batch()

        # There are 6 valid sequences in total, and we are using a batch size of 2.
        for i in range(3):
            query = [next_sequence_tensor, next_sequence_timing_tensor]
            next_sequence, next_sequence_timing = self.sess.run(query,
                                                                feed_dict=data_set.get_feed_dict())

            self.assertListEqual(next_sequence_timing[0].tolist(), [0.0, 0.5, 1.0])
            self.assertListEqual(next_sequence_timing[1].tolist(), [0.0, 0.5, 1.0])
            self.assertTupleEqual(np.shape(next_sequence), (2, 3, 264, 470, 3))

    def test_data_read_write_multi(self):
        """
        Tests for the case where multiple inbetween_location configs are provided.
        """
        data_set = InterpDataSet(self.data_directory, [[1], [1, 0, 0]], batch_size=1)
        data_set.preprocess_raw(shard_size=1)

        output_paths = data_set.get_tf_record_names()
        [self.assertTrue(os.path.isfile(output_path)) for output_path in output_paths]

        # For a shard size of 1 the number of files is the number of video shots (or the number of sub-folders).
        self.assertEqual(len(output_paths), 2)

        data_set.load(self.sess)

        next_sequence_tensor, next_sequence_timing_tensor = data_set.get_next_batch()

        # There are 6 + 4 valid sequences in total (different inbetweening locations),
        # and we are using a batch size of 1.
        # For the sparse sequence, the dog-agility shot is not long enough, so a Tensor of zeros will be returned.
        num_dense_sequences = 0
        num_sparse_sequences = 0
        for i in range(10):
            query = [next_sequence_tensor, next_sequence_timing_tensor]
            next_sequence, next_sequence_timing = self.sess.run(query,
                                                                feed_dict=data_set.get_feed_dict())

            if next_sequence_timing[0].tolist() == [0.0, 0.5, 1.0]:
                num_dense_sequences += 1
                is_sparse = False
            elif next_sequence_timing[0].tolist() == [0.0, 0.25, 1.0]:
                num_sparse_sequences += 1
                is_sparse = True

            self.assertTupleEqual(np.shape(next_sequence), (1, 3, 264, 470, 3))
            if VISUALIZE:
                if is_sparse:
                    print('Showing sparse sequence with timings [0.0, 0.2, 1.0] ...')
                else:
                    print('Showing dense sequence with timings [0.0, 0.5, 1.0] ...')
                for j in range(3):
                    show_image(next_sequence[0][j])

        self.assertEqual(num_dense_sequences, 6)
        self.assertEqual(num_sparse_sequences, 4)

    def tearDown(self):
        data_set = InterpDataSet(self.data_directory, [[1]], batch_size=2)
        output_paths = data_set.get_tf_record_names()
        for output_path in output_paths:
            if os.path.isfile(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
