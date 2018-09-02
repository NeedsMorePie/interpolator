import tensorflow as tf
import os
import shutil


class Interp:
    def __init__(self, saved_model_dir=None):
        """
        :param saved_model_dir: Str. Path to the SavedModel export directory.
        """
        self.saved_model_dir = saved_model_dir
        self.predict_fn = None

        # For creating the graph that the SavedModel will use.
        self.graph_created = False
        self.image_a = None
        self.image_b = None
        self.interpolated = None
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
        outputs = self._get_forward(image_a, image_b, t, reuse_variables=reuse_variables)

        # Create separate graph for SavedModel.
        if self.saved_model_dir is not None:
            self.image_a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.image_b = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.interpolated = self._get_forward(self.image_a, self.image_b, t, reuse_variables=tf.AUTO_REUSE)[0]
            self.graph_created = True
        return outputs

    def _get_forward(self, image_a, image_b, t, reuse_variables=tf.AUTO_REUSE):
        """
        See get_forward.
        """
        raise NotImplementedError

    def predict(self, images_a, images_b):
        """
        # TODO: Support variable time interpolation.
        :param image_a: Numpy array. Image of shape [batch_size, H, W, C]. The image at t = 0.
        :param image_b: Numpy array. Image of shape [batch_size, H, W, C]. The image at t = 1.
        :return: Numpy array of shape [batch_size, H, W, C]. The interpolation at t = 0.5.
        """
        assert self.saved_model_dir is not None
        predictions = self.predictor({
            'image_0': images_a,
            'image_1': images_b
        })
        return predictions['output']

    def load_saved_model(self):
        assert self.saved_model_dir is not None
        print('Loading SavedModel from: ', self.saved_model_dir)
        self.predictor = tf.contrib.predictor.from_saved_model(self.saved_model_dir)

    def save_saved_model(self, session, overwrite=False):
        """
        :param session: Tf Session.
        :param overwrite: Bool. Whether to overwrite an existing SavedModel if it exists.
        """
        assert self.saved_model_dir is not None
        assert self.graph_created
        inputs = {'image_0': self.image_a, 'image_1': self.image_b}
        outputs = {'output': self.interpolated}
        if overwrite and os.path.exists(self.saved_model_dir):
            print('Removing previous SavedModel from: ', self.saved_model_dir)
            shutil.rmtree(self.saved_model_dir)
        tf.saved_model.simple_save(session, self.saved_model_dir, inputs, outputs)
        print('SavedModel saved at: ', self.saved_model_dir)
