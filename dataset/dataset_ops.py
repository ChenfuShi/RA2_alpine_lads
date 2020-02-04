import logging
import os

import numpy as np
import tensorflow as tf

import dataset.image_ops as ops


AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_images(dataset, directory, update_labels = False):
    def __load(file, y):
        file_name = file[0]
        flip = file[1]
        
        flip_img = flip == 'Y'
        
        return ops.load_image(file_name, y, update_labels, directory, flip_img)

    return dataset.map(__load, num_parallel_calls=AUTOTUNE)

def shuffle_and_repeat_dataset(dataset, buffer_size = 200):
    dataset = dataset.shuffle(buffer_size = buffer_size)
    
    return dataset.repeat()

def cache_dataset(dataset, cache):
    if cache:
        if isinstance(cache, str):
            try:
                os.makedirs(os.path.expanduser(cache), exist_ok = True)
                dataset = dataset.cache(os.path.expanduser(cache))
            except FileNotFoundError:
                logging.warn("Missing permissions to create directory for caching!")

                pass  
        else:
            dataset = dataset.cache()
            
    return dataset

def batch_and_prefetch_dataset(dataset, batch_size = 128):
    dataset = dataset.batch(batch_size)
    
    return dataset.prefetch(buffer_size = AUTOTUNE)    

def resize_images(dataset, img_width, img_height, update_labels = False):
    def __resize(img, y):
        return ops.resize_image(img, y, update_labels, img_width, img_height)

    return dataset.map(__resize, num_parallel_calls=AUTOTUNE)

def randomly_augment_images(dataset, augments = [ops.random_rotation, ops.random_brightness_and_contrast, ops.random_crop], update_labels = False):
    def __clip_image(img, y):
        img = ops.clip_image(img)

        return img, y

    for aug in augments:
        dataset = _apply_random_augment(dataset, aug, update_labels)
        
    # After augmentations, scale values back to lie between 0 & 1
    return dataset.map(__clip_image, num_parallel_calls=AUTOTUNE)

def _apply_random_augment(dataset, aug, update_labels):
    def __apply_random_augment(img, y):
        return ops.apply_augment(img, y, aug, update_labels = update_labels)
    
    return dataset.map(__apply_random_augment, num_parallel_calls=AUTOTUNE)