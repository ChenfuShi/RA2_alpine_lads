import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging
import math

# Rotate image by ~10 degrees in either direction
RAND_ROTATION_MIN = -10
RAND_ROTATION_MAX = 10

scales = list(np.arange(0.8, 1.0, 0.01))
boxes = np.zeros((len(scales), 4))

for i, scale in enumerate(scales):
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes[i] = [x1, y1, x2, y2]

def random_brightness_and_contrast(img, y):
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, 0, 0.2)
    
    logging.info("Applied random_brightness_and_contrast")
    
    return img, y
    
def random_rotation(img, y):
    random_degree_angle = tf.random.uniform(shape=[], minval=RAND_ROTATION_MIN, maxval=RAND_ROTATION_MAX)
    
    img = tfa.image.rotate(img, _calc_radians_for_degrees(random_degree_angle))
    
    tf.print(random_degree_angle)
    
    return img, y

def random_crop(img, y):
    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=tf.shape(img[:, :, 0]))
    
    # Return a random crop
    crop = crops[tf.random.uniform([], minval=0, maxval=len(scales), dtype=tf.int32)]
    
    return crop, y

def _calc_radians_for_degrees(degree_angle):
    return degree_angle * math.pi / 180