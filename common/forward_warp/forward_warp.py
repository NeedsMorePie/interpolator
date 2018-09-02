import os.path
import tensorflow as tf
from sys import platform
from tensorflow.python.framework import ops


DISOCC_THRESH = 0.5


# Load op library.
if platform == 'win32':
    lib_path = os.path.join('build', 'forward_warp_op.dll')
else:
    lib_path = os.path.join('build', 'libforward_warp_op.so')
if os.path.isfile(lib_path):
    mod = tf.load_op_library(lib_path)
else:
    print('Warning: No CUDA implementation of forward_warp found. Falling back to the Tensorflow version.')
    mod = None


def is_forward_warp_cuda():
    """
    :return: Bool. Whether the forward_warp is using a custom CUDA op.
    """
    return mod is not None


def forward_warp(features, flow, splat_variance=0.5):
    """
    For an algorithm that gives the same end result, see section 3 in https://arxiv.org/pdf/1711.05890.pdf.
    Note that the actual implementation here is not n^2, and should be linear in GPU memory.
    :param features: A Tensor. Features to be warped, of shape [batch_size, H, W, C].
    :param flow: A Tensor. Un-normalized flow in image pixel units, of shape [batch_size, H, W, 2].
                 Flow vectors should have (x, y) ordering.
    :param splat_variance: Float. Variance of the splat. Only used for the CUDA op.
    """
    if is_forward_warp_cuda():
        return mod.forward_warp(features, flow, variance=splat_variance)
    else:
        return forward_warp_tf(features, flow)


if is_forward_warp_cuda():
    @ops.RegisterGradient('ForwardWarp')
    def _ForwardWarpGrad(op, grad):
        image_grad, flow_grad = mod.forward_warp_grad(
            grad, op.inputs[0], op.inputs[1], variance=op.get_attr('variance'))
        return [image_grad, flow_grad]


def forward_warp_tf(features, flow):
    """
    For an algorithm that gives the same end result, see section 3 in https://arxiv.org/pdf/1711.05890.pdf.
    Note that the actual implementation here is not n^2, and should be linear in GPU memory.
    :param features: A Tensor. Features to be warped, of shape [batch_size, H, W, C].
    :param flow: A Tensor. Un-normalized flow in image pixel units, of shape [batch_size, H, W, 2].
                 Flow vectors should have (x, y) ordering.
    """
    with tf.name_scope('forward_warp'):

        # Flip (x, y) to (y, x) for flow, to avoid further confusion.
        flow = tf.reverse(flow, axis=[-1])

        # Get target indices along with corresponding values to add.
        indices, splat_values = _get_translated_pixels(features, flow)

        # Mask out out-of-bounds splat values
        height, width, channels = tf.shape(features)[1], tf.shape(features)[2], tf.shape(features)[3]
        greater_than_zero = tf.greater_equal(indices, 0)
        greater_than_zero = tf.cast(tf.reduce_all(greater_than_zero, axis=-1), tf.float32)
        less_than_height = tf.cast(tf.less(indices[..., 0], height), tf.float32)
        less_than_width = tf.cast(tf.less(indices[..., 1], width), tf.float32)
        within_bounds = greater_than_zero * less_than_height * less_than_width
        splat_values *= tf.expand_dims(within_bounds, axis=-1)

        # Add batch indices, clip xy indices, and flatten.
        batch_size = tf.shape(features)[0]
        num_indices = tf.shape(indices)[1]
        batch_indices = tf.range(0, batch_size)
        batch_indices = tf.tile(batch_indices, [num_indices])
        batch_indices = tf.reshape(batch_indices, (num_indices, batch_size))
        batch_indices = tf.transpose(batch_indices)
        y_indices = tf.clip_by_value(indices[..., 0], 0, height-1)
        x_indices = tf.clip_by_value(indices[..., 1], 0, width-1)
        indices = tf.stack([batch_indices, y_indices, x_indices], axis=-1)

        # Scatter into output.
        warped = tf.scatter_nd(indices, splat_values, tf.shape(features))
        return warped


def _get_translated_pixels(features, translations):
    """
    :param features: A Tensor. Of shape [batch_size, H, W, C].
    :param translations: A Tensor. Translations in image pixel units, of shape [batch_size, H, W, 2].
    :return: indices: Tensor of shape [batch_size, num_indices, 2]. The indices to target.
             values: Tensor of shape [batch_size, num_indices, C]. The values to put at the corresponding indices.
    """
    with tf.name_scope('translate_pixels'):

        # Get translated mesh-grid.
        fshape = tf.shape(features)
        batch_size, height, width, channels = fshape[0], fshape[1], fshape[2], fshape[3]
        x_1d, y_1d = tf.range(0, width), tf.range(0, height)
        x_2d, y_2d = tf.meshgrid(x_1d, y_1d)
        indices_2d = tf.cast(tf.stack([y_2d, x_2d], axis=-1), tf.float32)
        translated_indices = translations + indices_2d

        # Get splat corners.
        floored = tf.cast(tf.floor(translated_indices), tf.int32)
        ceiled = floored + 1
        tl_indices = floored
        tr_indices = tf.stack([floored[..., 0], ceiled[..., 1]], axis=-1)
        br_indices = ceiled
        bl_indices = tf.stack([ceiled[..., 0], floored[..., 1]], axis=-1)

        # Compute splat values, using inverse bi-linear interpolation formula.
        tl_diff = tf.expand_dims(tf.abs(tf.cast(tl_indices, tf.float32) - translated_indices), axis=-1)
        tr_diff = tf.expand_dims(tf.abs(tf.cast(tr_indices, tf.float32) - translated_indices), axis=-1)
        br_diff = tf.expand_dims(tf.abs(tf.cast(br_indices, tf.float32) - translated_indices), axis=-1)
        bl_diff = tf.expand_dims(tf.abs(tf.cast(bl_indices, tf.float32) - translated_indices), axis=-1)
        tl_vals = features * (1.0 - tl_diff[..., 0, :]) * (1.0 - tl_diff[..., 1, :])
        tr_vals = features * (1.0 - tr_diff[..., 0, :]) * (1.0 - tr_diff[..., 1, :])
        br_vals = features * (1.0 - br_diff[..., 0, :]) * (1.0 - br_diff[..., 1, :])
        bl_vals = features * (1.0 - bl_diff[..., 0, :]) * (1.0 - bl_diff[..., 1, :])

        # Combine and flatten shape.
        all_indices = tf.stack([tl_indices, tr_indices, br_indices, bl_indices], axis=1)
        all_vals = tf.stack([tl_vals, tr_vals, br_vals, bl_vals], axis=1)
        all_indices = tf.reshape(all_indices, (batch_size, -1, 2))
        all_vals = tf.reshape(all_vals, (batch_size, -1, channels))
        return all_indices, all_vals


def create_disocclusion_mask(flow, splat_variance=1.0):
    """
    Creates a disocclusion mask representing areas that were previously occluded and will become visible.
    This is done by forward warping some ones and thresholding them for visibility.

    Disocclusion maps are used by the UnFlow loss:
    https://github.com/simonmeister/UnFlow/blob/8bff4939963c7d0adb9435880dc506fb3f988080/src/e2eflow/core/losses.py#L28
    This isn't mentioned in the paper anywhere, but clearly enough, it is in the code.
    :param flow: Tensor of shape [B, H, W, 2].
    :param splat_variance: Float. Variance of the splat. Only used for the CUDA op.
    :return: Tensor of shape [B, H, W, 1].
    """
    with tf.name_scope('disocclusion_mask'):
        batch, height, width, _ = tf.unstack(tf.shape(flow))
        prewarp_mask = tf.ones([batch, height, width, 1], dtype=tf.float32)
        forward_warped_mask = forward_warp(prewarp_mask, flow, splat_variance=splat_variance)
        return tf.cast(forward_warped_mask < DISOCC_THRESH, dtype=tf.float32)
