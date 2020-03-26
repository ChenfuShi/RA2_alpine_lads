import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging
import math
import dataset.ops.landmark_ops as lm_ops

PNG_EXTENSION_REGEX = '(?i).*png'
JPG_EXTENSION_REGEX = '(?i).*jp[e]?g'

def _create_boxes(scales = np.arange(0.8, 1, 0.01)):
    boxes = np.zeros((scales.size, 4))
    
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        
        boxes[i] = [x1, y1, x2, y2]
        
    return boxes

def load_image(file_info, y, directory, update_labels = False, imagenet = False):
    file_name = file_info[0]
    file_type = file_info[1]
    flip_img = file_info[2] == 'Y'

    img = tf.io.read_file(directory + '/' + file_name + '.' + file_type)
    if imagenet:
        img = tf.io.decode_image(img, channels = 3, dtype = tf.float32)
    else:
        img = tf.io.decode_image(img, channels = 1, dtype = tf.float32)

    if flip_img:    
        img = tf.image.flip_left_right(img)

        if update_labels:
            img_shape = tf.shape(img)
            
            y = lm_ops.flip_landmarks(y, img_shape)

    return img, y

def resize_image(img, y, img_height, img_width, pad_resize = True, update_labels = False):
    old_shape = tf.shape(img)
    
    if pad_resize:
        img = tf.image.resize_with_pad(img, img_height, img_width)
    else:
        img = tf.image.resize(img, (img_height, img_width))

    if(update_labels):
        new_shape = tf.shape(img)

        y = lm_ops.resize_landmarks(y, old_shape, new_shape)

    return img, y

def apply_augment(img, y, aug, update_labels = False, cutoff = 0.1):
    img, y = tf.cond(tf.random.uniform([], 0, 1) > cutoff, lambda: aug(img, y, update_labels), lambda: (img, y))

    return img, y

def clip_image(img):
    img = tf.clip_by_value(img, 0, 1)

    return img

def random_brightness_and_contrast(img, y, update_labels, max_delta = 0.2, max_contrast = 0.2):
    img = tf.image.random_brightness(img, max_delta = max_delta)
    img = tf.image.random_contrast(img, 1 - max_contrast, 1 + max_contrast)
    
    return img, y

def random_rotation(img, y, update_labels, min_rot = -20, max_rot = 20):
    random_degree_angle = tf.random.uniform(shape=[], minval = min_rot, maxval = max_rot)
    
    radian_angle = _calc_radians_for_degrees(random_degree_angle)

    img = tfa.image.rotate(img, radian_angle)

    if(update_labels):
        y = lm_ops.rotate_landmarks(y, tf.shape(img), radian_angle)
    
    return img, y

def random_crop(img, y, update_labels, boxes = _create_boxes()):
    no_boxes = boxes.shape[0]
    
    original_img_shape = tf.shape(img)

    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes = boxes, box_indices = np.zeros(no_boxes), crop_size = tf.shape(img[:, :, 0]))
    
    random_box_idx = tf.random.uniform([], minval = 0, maxval = no_boxes, dtype = tf.int32)

    # Return a random crop
    img = crops[random_box_idx]
    
    if(update_labels):
        y = lm_ops.crop_landmarks(y, tf.convert_to_tensor(boxes)[random_box_idx], original_img_shape)

    return img, y

def random_gaussian_noise(img, y, update_labels, noise_strength = 5):
    noise = tf.random.normal(shape = tf.shape(img), stddev = (noise_strength / 255), dtype = tf.float32)
    noise_img = img + noise
    noise_img = clip_image(noise_img)
        
    return noise_img, y

def _calc_radians_for_degrees(degree_angle):
    return degree_angle * math.pi / 180


def get_3_channels(img,y):
    return tf.image.grayscale_to_rgb(img), y
