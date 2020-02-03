import numpy as np
import tensorflow as tf

def flip_landmarks(y, shape):
    shape = tf.cast(shape, dtype = tf.float64)

    updates_idx = tf.reshape(tf.range(tf.shape(y)[0])[0::2], (6, 1))
    updates = tf.math.abs(tf.math.subtract(shape[1], y[0::2]))
    
    return tf.tensor_scatter_nd_update(y, updates_idx, updates)