import logging
import math

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import dataset.ops.landmark_ops as lm_ops

PNG_EXTENSION_REGEX = '(?i).*png'
JPG_EXTENSION_REGEX = '(?i).*jp[e]?g'

def _create_boxes(min_scale = 0.8):
    scales = np.arange(min_scale, 1, 0.01)
    
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
    
    channels = 1
    
    if imagenet:
        channels = 3
    
    img = tf.io.decode_image(img, channels = channels, dtype = tf.float32)

    if flip_img:    
        img = tf.image.flip_left_right(img)

        if update_labels:
            img_shape = tf.shape(img)
            
            y = lm_ops.flip_landmarks(y, img_shape)

    return img, y

def clahe_img(img, clip_limit = 2., grid_size = 8):
    # Open CV requires uint8
    img_array = tf.image.convert_image_dtype(img, dtype = 'uint8')
        
    clahe_img = tf.py_function(_clahe_img, [img_array, clip_limit, grid_size], tf.uint8)
    
    # OpenCV removes the last channel, so add it back and then convert back to float
    clahe_img = tf.expand_dims(clahe_img, -1)
    clahe_img = tf.image.convert_image_dtype(clahe_img, dtype = tf.float32)
    
    clahe_img.set_shape(img.get_shape())
    
    return clahe_img

def _clahe_img(img_array, clip_limit, grid_size):
    clahe = cv.createCLAHE(clipLimit = clip_limit, tileGridSize = (grid_size, grid_size))
    clahe_img = clahe.apply(img_array.numpy())
    
    return clahe_img

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

def apply_augment(img, y, aug, p, update_labels = False):
    img, y = tf.cond(tf.random.uniform([], 0, 1) < p, lambda: aug(img, y, update_labels), lambda: (img, y))

    return img, y

def clip_image(img):
    img = tf.clip_by_value(img, 0, 1)

    return img

def create_augments(augments):
    return list(map(create_augment, augments))

def create_augment(augment):
    aug  = augment['augment']
    p = augment.get('p', 0.9)
    params = augment.get('params', {})

    augment_function = aug(**params)

    def _augment(img, y, update_labels):
        img, y = tf.cond(tf.random.uniform([], 0, 1) < p, lambda: augment_function(img, y, update_labels), lambda: (img, y))

        return img, y

    return _augment

def clahe_aug(clip_limit = 2., grid_size = 8):
    def _clahe(img, y, update_labels):
        img = clahe_img(img, clip_limit = clip_limit, grid_size = grid_size)

        return img, y

    return _clahe

def random_flip(flip_right_left = True, flip_up_down = True):
    def _random_flip(img, y, update_labels):
        if flip_right_left:
            img = tf.image.random_flip_left_right(img)

        if flip_up_down:
            img = tf.image.random_flip_up_down(img)
    
        if update_labels is True:
            logging.error('Update labels not available for random_flip!')
    
        return img, y

    return _random_flip

def random_brightness_and_contrast(max_delta = 0.1, max_contrast = 0.1):
    def _random_brightness_and_contrast(img, y, update_labels):
        img = tf.image.random_brightness(img, max_delta = max_delta)
        img = tf.image.random_contrast(img, 1 - max_contrast, 1 + max_contrast)
        
        return img, y

    return _random_brightness_and_contrast

def random_rotation(max_rot = 20):
    min_rot = -1 * max_rot 
    
    def _random_rotation(img, y, update_labels):
        random_degree_angle = tf.random.uniform(shape=[], minval = min_rot, maxval = max_rot)
        
        radian_angle = _calc_radians_for_degrees(random_degree_angle)

        img = tfa.image.rotate(img, radian_angle)

        if(update_labels):
            y = lm_ops.rotate_landmarks(y, tf.shape(img), radian_angle)
        
        return img, y

    return _random_rotation

def random_crop(min_scale = 0.8):
    boxes = _create_boxes(min_scale = min_scale)
    
    def _random_crop(img, y, update_labels):
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

    return _random_crop

def random_gaussian_noise(max_noise_strength = 3):
    def _random_gaussian_noise(img, y, update_labels):
        noise_factor = tf.random.uniform([], minval = 1e-6, maxval = max_noise_strength)
        noise = tf.random.normal(shape = tf.shape(img), stddev = (noise_factor / 255), dtype = tf.float32)
        noise_img = img + noise
        noise_img = clip_image(noise_img)
        
        return noise_img, y

    return _random_gaussian_noise

def _calc_radians_for_degrees(degree_angle):
    return degree_angle * math.pi / 180

def get_3_channels(img,y):
    return tf.image.grayscale_to_rgb(img), y
