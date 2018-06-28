import tensorflow as tf
from context_interp.vgg19_features.model.model import Vgg19


class Vgg19Features:

    def load_pretrained_weights(self):
        vgg19_data = Vgg19.load_params_dict(load_small=True)
        self.vgg19 = Vgg19(data_dict=vgg19_data)

    def get_context_features(self, images):
        """
        :param images: A Tensor. Of shape [batch, H, W, num_features].
        :return: A Tensor. Of shape [batch, H, W, 64].
        """
        image_features, _ = self.vgg19.build_up_to_conv1_2(images, trainable=False)
        return image_features

    def get_perceptual_features(self, images):
        """
        :param images: A Tensor. Of shape [batch, H, W, num_features].
        :return: A Tensor. Of shape [batch, H / 8, W / 8, 512].
        """
        image_features, _ = self.vgg19.build_up_to_conv4_4(images, trainable=False)
        return image_features
