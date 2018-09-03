import numpy as np
import tensorflow as tf
import unittest
import os
import shutil
from interp.interp import Interp
from data.interp.interp_data import InterpDataSet
from data.interp.davis.davis_preprocessor import DavisDataSetPreprocessor
from common.utils.tf import print_tensor_shape


# Mock class for testing purposes.
class MockInterp(Interp):
    def __init__(self, saved_model_dir):
        super().__init__('mock', saved_model_dir=saved_model_dir)

    def _get_forward(self, images_0, images_1, t, reuse_variables=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            a = tf.layers.conv2d(images_0, 3, (1, 1))
            b = images_1
            sum = a + b
        return sum, None


class TestInterpSavedModel(unittest.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Define paths.
        cur_dir = os.path.dirname(__file__)
        self.saved_model_directory = os.path.join(cur_dir, 'saved_model')
        self.data_directory = os.path.join(cur_dir, 'test_data')
        self.tf_record_directory = os.path.join(self.data_directory, 'tfrecords')

    def test_saved_model(self):

        # Create dataset.
        data_set = InterpDataSet(self.tf_record_directory, [[1]], batch_size=1)
        preprocessor = DavisDataSetPreprocessor(self.tf_record_directory, [[1]], shard_size=1)
        preprocessor.preprocess_raw(self.data_directory)

        output_paths = data_set.get_tf_record_names()
        [self.assertTrue(os.path.isfile(output_path)) for output_path in output_paths]
        self.assertEqual(1, len(output_paths))
        data_set.load(self.sess)
        next_sequence_tensor, next_sequence_timing_tensor = data_set.get_next_batch()

        model = MockInterp(saved_model_dir=self.saved_model_directory)
        interpolated_tensor, _ = model.get_forward(next_sequence_tensor[:, 0, ...], next_sequence_tensor[:, 2, ...], 0.5)
        self.sess.run(tf.global_variables_initializer())
        self.assertEquals(2, len(tf.trainable_variables()))

        # Check variables.
        values = self.sess.run(tf.trainable_variables())
        sums = np.array([np.sum(value) for value in values])
        self.assertNotAlmostEquals(0.0, np.sum(sums))

        interpolated, next_sequence = self.sess.run([interpolated_tensor, next_sequence_tensor],
                                                    feed_dict=data_set.get_train_feed_dict())

        self.assertTupleEqual((1, 256, 256, 3), np.shape(interpolated))
        self.assertTupleEqual((1, 3, 256, 256, 3), np.shape(next_sequence))
        self.assertNotAlmostEquals(0.0, np.sum(interpolated))

        # Save to SavedModel.
        model.save_saved_model(self.sess, overwrite=True)
        interpolated_from_saved_model = model.interpolate_from_saved_model(next_sequence[:, 0, ...],
                                                                           next_sequence[:, 2, ...])

        # Check that inference results are the same as during training.
        self.assertTupleEqual((1, 256, 256, 3), np.shape(interpolated_from_saved_model))
        self.assertEquals(interpolated.tolist(), interpolated_from_saved_model.tolist())
        
    def tearDown(self):
        dirs_to_remove = [self.tf_record_directory, self.saved_model_directory]
        for dir in dirs_to_remove:
            if os.path.exists(dir):
                shutil.rmtree(dir)

