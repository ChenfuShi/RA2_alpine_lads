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

def _randomly_augment_dataset(dataset):
    for aug in self.augments:
        dataset = _apply_random_augment(dataset, aug)

    # After augmentations, scale values back to lie between 0 & 1
    dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=AUTOTUNE)

    return dataset

def _apply_random_augment(dataset, aug, cutoff = 0.75):
    # Randomly apply each augmentation 1 - cutoff% of the time
    return dataset.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > cutoff, lambda: aug(x), lambda: x), num_parallel_calls=AUTOTUNE)

def random_brightness_and_contrast(img, y):
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, 0, 0.2)
    
    logging.info("Applied random_brightness_and_contrast")
    
    return img, y
    
def random_rotation(img, y):
    random_degree_angle = tf.random.uniform(shape=[], minval=RAND_ROTATION_MIN, maxval=RAND_ROTATION_MAX)
    
    img = tfa.image.rotate(img, _calc_radians_for_degrees(random_degree_angle))
    
    logging.info("Applied random_rotation: " + str(random_degree_angle))
    
    return img, y

def random_crop(img, y):
    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(128, 128))
    
    # Return a random crop
    crop = crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
    
    return crop, y

def _calc_radians_for_degrees(degree_angle):
    return degree_angle * math.pi / 180