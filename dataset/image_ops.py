import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging
import math

def _create_boxes(scales = np.arange(0.8, 1, 0.01)):
    boxes = np.zeros((scales.size, 4))
    
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        
        boxes[i] = [x1, y1, x2, y2]
        
    return boxes

def load_image(file, directory, flip_str):
    file_path = directory + "/" + file + ".jpg"
    
    img = tf.io.read_file(file_path)    
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
            
    if flip_str in str(file):
        img = tf.image.flip_left_right(img)
                
    return img

def resize_image(img, img_width, img_height):
    img = tf.image.resize(img, [ img_height, img_width])

    return img

def apply_augment(img, aug, cutoff = 0.6):
    img = tf.cond(tf.random.uniform([], 0, 1) > cutoff, lambda: aug(img), lambda: img)

    return img

def clip_image(img):
    img = tf.clip_by_value(img, 0, 1)

    return img

def random_brightness_and_contrast(img, max_delta = 0.2, max_contrast = 0.2):
    img = tf.image.random_brightness(img, max_delta=max_delta)
    img = tf.image.random_contrast(img, 0, max_contrast)
    
    return img
    
def random_rotation(img, min_rot = -10, max_rot = 10):
    random_degree_angle = tf.random.uniform(shape=[], minval=min_rot, maxval=max_rot)
    
    img = tfa.image.rotate(img, _calc_radians_for_degrees(random_degree_angle))
    
    return img

def random_crop(img, boxes = _create_boxes()):
    no_boxes = boxes.shape[0]
    
    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(no_boxes), crop_size=tf.shape(img[:, :, 0]))
    
    # Return a random crop
    img = crops[tf.random.uniform([], minval=0, maxval=no_boxes, dtype=tf.int32)]
    
    return img

def _calc_radians_for_degrees(degree_angle):
    return degree_angle * math.pi / 180

