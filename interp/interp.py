import tensorflow as tf
import os
import shutil


class Interp:
    def __init__(self, name, saved_model_dir=None):
        """
        :param name: Str. For Tf variable scoping.
        :param saved_model_dir: Str. Path to the SavedModel export directory.
        """
        self.name = name
        self.saved_model_dir = saved_model_dir
        self.predictor = None

        # For creating the graph that the SavedModel will use.
        self.graph_created = False
        self.images_0 = None
        self.images_1 = None
        self.interpolated = None
        # TODO: make this part of the interpolation args.
        self.t = 0.5

    def get_forward(self, images_0, images_1, t, reuse_variables=tf.AUTO_REUSE):
        """
        :param images_0: Tensor of shape [batch_size, H, W, 3].
        :param images_1: Tensor of shape [batch_size, H, W, 3].
        :param t: Float. Specifies the interpolation point (i.e 0 for images_0, 1 for images_1).
        :return: interpolated: The interpolated image. Tensor of shape [batch_size, H, W, 3].
                 others: There may be other arguments after the first (e.g interpolated, ret1, ret2).
                         If not, this field must be left as None.
        """
        outputs = self._get_forward(images_0, images_1, t, reuse_variables=reuse_variables)
        assert isinstance(outputs, tuple)

        # Create separate graph for SavedModel.
        if self.saved_model_dir is not None:
            self.images_0 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.images_1 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            save_outputs = self._get_forward(self.images_0, self.images_1, t, reuse_variables=tf.AUTO_REUSE)
            self.interpolated = save_outputs[0]
            self.graph_created = True
        return outputs

    def _get_forward(self, images_0, images_1, t, reuse_variables=tf.AUTO_REUSE):
        """
        Subclasses should implement this method. See get_forward.
        """
        raise NotImplementedError

    def interpolate_from_saved_model(self, images_0, images_1):
        """
        # TODO: Support variable time interpolation.
        :param images_0: Numpy array. Image of shape [batch_size, H, W, C]. The image at t = 0.
        :param images_1: Numpy array. Image of shape [batch_size, H, W, C]. The image at t = 1.
        :return: Numpy array of shape [batch_size, H, W, C]. The interpolation at t = 0.5.
        """
        assert self.saved_model_dir is not None
        if self.predictor is None:
            self.load_saved_model()
        predictions = self.predictor({
            'images_0': images_0,
            'images_1': images_1
        })
        return predictions['output']

    def load_saved_model(self):
        assert self.saved_model_dir is not None
        print('Loading SavedModel from: ', self.saved_model_dir)
        self.predictor = tf.contrib.predictor.from_saved_model(self.saved_model_dir)

    def save_saved_model(self, session, overwrite=False):
        """
        :param session: Tf Session.
        :param overwrite: Bool. Whether to remove and overwrite an existing SavedModel directory if it exists.
        """
        assert self.saved_model_dir is not None
        assert self.graph_created
        inputs = {'images_0': self.images_0, 'images_1': self.images_1}
        outputs = {'output': self.interpolated}
        if os.path.exists(self.saved_model_dir):
            if not overwrite:
                raise AttributeError
            print('Removing previous SavedModel from: ', self.saved_model_dir)
            shutil.rmtree(self.saved_model_dir)
        print(self.interpolated.get_shape().as_list())
        tf.saved_model.simple_save(session, self.saved_model_dir, inputs, outputs)
        print('SavedModel saved at: ', self.saved_model_dir)
