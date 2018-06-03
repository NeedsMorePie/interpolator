import tensorflow as tf


def forward_warp(features, flow, max_image_area=1280*720):
    """
    For an algorithm that gives the same end result, see section 3 in https://arxiv.org/pdf/1711.05890.pdf.
    Note that the actual implementation here is not n^2, and should be linear in GPU memory.
    :param features: A Tensor. Features to be warped, of shape [batch_size, H, W, C].
    :param flow: A Tensor. Un-normalized flow in image pixel units, of shape [batch_size, H, W, 2].
    :param max_image_area: The maximum value for width * height of the input features. It is ok to specify a value
                           larger than the actual area (at the expense of performance), but it cannot be any less.
    """

    # Get target indices along with corresponding values to add.

    # Partition based on target index.

    # Aggregate for each index.

    # Scatter into output.


def get_translated_pixels(features, translations):
    """
    :param features: A Tensor. Of shape [batch_size, H, W, C].
    :param translations: A Tensor. Translations in image pixel units, of shape [batch_size, H, W, 2].
    :return: indices: Tensor of shape [batch_size, num_indices, 2]. The indices to target.
             values: Tensor of shape [batch_size, num_indices, C]. The values to put at the corresponding indices.
    """

    # Get translated mesh-grid.
    fshape = tf.shape(features)
    batch_size, height, width, channels = fshape[0], fshape[1], fshape[2], fshape[3]
    x_1d, y_1d = tf.range(0, width), tf.range(0, height)
    x_2d, y_2d = tf.meshgrid(x_1d, y_1d)
    indices_2d = tf.cast(tf.stack([y_2d, x_2d], axis=-1), tf.float32)
    translated_indices = translations + indices_2d

    # Get splat corners.
    ceiled = tf.cast(tf.ceil(translated_indices), tf.int32)
    floored = tf.cast(tf.floor(translated_indices), tf.int32)
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

    # Zero out splat values for everything other than the first one,
    # if target has integer coordinates (ceil and floor are the same).
    # Otherwise we get incorrect duplication.
    tl_indices_compare = tf.cast(tl_indices, tf.float32)
    not_duplicated = 1.0 - tf.cast(tf.equal(tl_indices_compare, translated_indices), tf.float32)
    tr_vals *= not_duplicated
    br_vals *= not_duplicated
    bl_vals *= not_duplicated

    # Combine and flatten shape.
    all_indices = tf.stack([tl_indices, tr_indices, br_indices, bl_indices], axis=0)
    all_vals = tf.stack([tl_vals, tr_vals, br_vals, bl_vals], axis=0)
    all_indices = tf.reshape(all_indices, (batch_size, -1, 2))
    all_vals = tf.reshape(all_vals, (batch_size, -1, channels))
    return all_indices, all_vals