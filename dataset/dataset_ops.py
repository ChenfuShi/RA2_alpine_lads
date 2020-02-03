import numpy as np
import tensorflow as tf

import dataset.image_ops as ops
import dataset.landmark_ops as lm_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_images(dataset, directory, update_labels, flip_str = ""):
    def __load(file, y):
        file_name = file[0]
        flip = file[1]
        
        flip_img = flip == 'Y'
        
        img = ops.load_image(file_name, directory, flip_img)
        
        if flip_img:
            if update_labels:
                img_shape = tf.shape(img)
            
                y = lm_ops.flip_landmarks(y, img_shape)
            
        return img, y

    return dataset.map(__load, num_parallel_calls=AUTOTUNE)

def resize_images(dataset, img_width, img_height):
    def __resize(img, y):
        img = ops.resize_image(img, img_width, img_height)

        return img, y

    return dataset.map(__resize, num_parallel_calls=AUTOTUNE)

def randomly_augment_images(dataset, augments = [ops.random_rotation, ops.random_brightness_and_contrast, ops.random_crop]):
    def __clip_image(img, y):
        img = ops.clip_image(img)

        return img, y

    for aug in augments:
        dataset = _apply_random_augment(dataset, aug)
        
    # After augmentations, scale values back to lie between 0 & 1
    return dataset.map(__clip_image, num_parallel_calls=AUTOTUNE)

def _apply_random_augment(dataset, aug):
    def __apply_random_augment(img, y):
        img = ops.apply_augment(img, aug)

        return img, y
    
    return dataset.map(__apply_random_augment, num_parallel_calls=AUTOTUNE)