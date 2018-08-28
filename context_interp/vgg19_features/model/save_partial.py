import tensorflow as tf
from context_interp.vgg19_features.model.model import Vgg19
import os

# Usage of this script:
# Download the VGG19 NPY file from https://github.com/machrisaa/tensorflow-vgg.
# Put it wherever the variable data_path looks.
# After running this script a partial model (up to layer conv4_4) will be saved to output_path.
# The full model is ~574.7MB, the partial one ~64MB.
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_dir, 'vgg19.npy')
output_path = os.path.join(cur_dir, 'vgg19_conv4_4.npy')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load completely.
vgg19 = Vgg19(vgg19_npy_path=data_path)

# Build partially.
image_input = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
vgg19.build_up_to_conv4_4(image_input)
sess.run(tf.global_variables_initializer())

# This only saves whatever variables were created from build.
vgg19.save_npy(sess, npy_path=output_path)

