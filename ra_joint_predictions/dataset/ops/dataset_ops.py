import logging
import os

import numpy as np
import tensorflow as tf

import dataset.ops.image_ops as img_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_images(dataset, directory, update_labels = False):
    def __load(file_info, y):
        return img_ops.load_image(file_info, y, directory, update_labels = update_labels)

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

def resize_images(dataset, img_height, img_width, update_labels = False, pad_resize = True):
    def __resize(img, y):
        return img_ops.resize_image(img, y, img_height, img_width, pad_resize = pad_resize, update_labels = update_labels)

    return dataset.map(__resize, num_parallel_calls=AUTOTUNE)

def randomly_augment_images(dataset, augments = [img_ops.random_rotation, img_ops.random_brightness_and_contrast, img_ops.random_crop], update_labels = False):
    def __clip_image(img, y):
        img = img_ops.clip_image(img)

        return img, y

    for aug in augments:
        dataset = _apply_random_augment(dataset, aug, update_labels)
        
    # After augmentations, scale values back to lie between 0 & 1
    return dataset.map(__clip_image, num_parallel_calls=AUTOTUNE)

def _apply_random_augment(dataset, aug, update_labels):
    def __apply_random_augment(img, y):
        return img_ops.apply_augment(img, y, aug, update_labels = update_labels)
    
    return dataset.map(__apply_random_augment, num_parallel_calls=AUTOTUNE)


def get_3_channels(dataset):
    def __get_3_channels(img, y):
        return img_ops.get_3_channels(img, y)
    
    return dataset.map(__get_3_channels, num_parallel_calls=AUTOTUNE)