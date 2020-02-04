import numpy as np
import tensorflow as tf

def flip_landmarks(y, shape):
    shape = tf.cast(shape, dtype = tf.float64)

    updates = shape[1] - y[0::2]
    
    updates_idx, _ = _get_update_idx(y)
    return tf.tensor_scatter_nd_update(y, updates_idx, updates)

def rotate_landmarks(y, shape, radians):
    radians = tf.cast(-1 * radians, dtype = tf.float64)

    x_coord_updates_idx, y_coord_updates_idx = _get_update_idx(y)

    sin_angle = tf.sin(radians)
    cos_angle = tf.cos(radians)

    m_y = shape[0] / 2
    m_x = shape[1] / 2

    y_coord_updates = (y[0::2] - m_x) * sin_angle + (y[1::2] - m_y) * cos_angle + m_y
    x_coord_updates = (y[0::2] - m_x) * cos_angle - (y[1::2] - m_y) * sin_angle + m_x

    updated_y = tf.tensor_scatter_nd_update(y, x_coord_updates_idx, x_coord_updates)
    updated_y = tf.tensor_scatter_nd_update(updated_y, y_coord_updates_idx, y_coord_updates)

    return updated_y

def resize_landmarks(y, old_shape, new_shape):
    old_shape = tf.cast(old_shape, dtype = tf.float64)
    new_shape = tf.cast(new_shape, dtype = tf.float64)

    x_coord_updates_idx, y_coord_updates_idx = _get_update_idx(y)
    
    x_ratio = new_shape[1] / old_shape[1]
    y_ratio = new_shape[0] / old_shape[0]

    inverse_x_ratio = 1 - x_ratio
    inverse_y_ratio = 1 - y_ratio

    if inverse_x_ratio > inverse_y_ratio:
        x_coord_updates = x_ratio * y[0::2]

        y_coord_updates = x_ratio * y[1::2]
        scaling_correction = old_shape[0] * x_ratio

        y_coord_updates = y_coord_updates + (new_shape[0] - scaling_correction) /  2
    else:
        y_coord_updates = y_ratio * y[1::2]

        x_coord_updates = y_ratio * y[0::2]
        scaling_correction = old_shape[1] * y_ratio

        x_coord_updates = x_coord_updates + (new_shape[1] - scaling_correction) / 2


    updated_y = tf.tensor_scatter_nd_update(y, x_coord_updates_idx, x_coord_updates)
    updated_y = tf.tensor_scatter_nd_update(updated_y, y_coord_updates_idx, y_coord_updates)

    return updated_y

def crop_landmarks(y, box, original_img_shape):
    original_img_shape = tf.cast(original_img_shape, dtype = tf.float64)

    inner_image_shape = original_img_shape * (tf.cast(1, dtype = tf.float64) - 2 * box[0])

    # Shift points in inner image first
    shift = (original_img_shape - inner_image_shape) / 2
    x_coord_updates = y[0::2] - shift[1]
    y_coord_updates = y[1::2] - shift[0]

    x_coord_updates_idx, y_coord_updates_idx = _get_update_idx(y)

    updated_y = tf.tensor_scatter_nd_update(y, x_coord_updates_idx, x_coord_updates)
    updated_y = tf.tensor_scatter_nd_update(updated_y, y_coord_updates_idx, y_coord_updates)

    # Then resize to new size
    return resize_landmarks(updated_y, inner_image_shape, original_img_shape)

def _get_update_idx(y):
    n_y = tf.shape(y)[0]

    idx_range = tf.range(n_y)
    idx_size = n_y / 2

    x_idx = tf.reshape(idx_range[0::2], (idx_size, 1))
    y_idx = tf.reshape(idx_range[1::2], (idx_size, 1))

    return x_idx, y_idx