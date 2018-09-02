import tensorflow as tf
import os
import shutil
from tensorflow.contrib import predictor


class Interp:
    def __init__(self, saved_model_dir):
        """
        :param saved_model_dir: Str. Path to the SavedModel export directory.
        """
        self.saved_model_dir = saved_model_dir
        self.predict_fn = None
        self.graph_created = False

        # TODO: make this part of the interpolation args.
        self.t = 0.5

    def get_forward(self, image_a, image_b, t, reuse_variables=tf.AUTO_REUSE):
        """
        :param image_a: Tensor of shape [batch_size, H, W, 3].
        :param image_b: Tensor of shape [batch_size, H, W, 3].
        :param t: Float. Specifies the interpolation point (i.e 0 for image_a, 1 for image_b).
        :return: interpolated: The interpolated image. Tensor of shape [batch_size, H, W, 3].
                 others: There may be other arguments after the first (e.g interpolated, ret1, ret2).
        """
        raise NotImplementedError

    def predict(self, images_a, images_b):
        """
        # TODO: Support variable time interpolation.
        :param image_a: Numpy array. Image of shape [batch_size, H, W, C]. The image at t = 0.
        :param image_b: Numpy array. Image of shape [batch_size, H, W, C]. The image at t = 1.
        :return: Numpy array of shape [batch_size, H, W, C]. The interpolation at t = 0.5.
        """
        predictions = predict_fn()

    def load_model(self):
        self.predict_fn = predictor.from_saved_model(self.saved_model_dir)

    def save_model(self, overwrite=False):
        """
        :param overwrite: Bool. Whether to overwrite an existing SavedModel if it exists.
        """
        if not self.graph_created:
            self._create_graph()

        inputs = {'image_0': self.image_a, 'image_1': self.image_b}
        outputs = {'output': self.interpolated}
        if overwrite and os.path.exists(self.saved_model_dir):
            print('Removing previous saved model ...')
            shutil.rmtree(self.saved_model_dir)
        tf.saved_model.simple_save(self.session, self.saved_model_dir, inputs, outputs)
        print('Saved model saved in path: ', self.saved_model_dir)

    def _create_graph(self):
        self.image_a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.image_b = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        outputs = self.get_forward(self.image_a, self.image_b, self.t)
        self.interpolated = outputs[0]
        self.graph_created = True
