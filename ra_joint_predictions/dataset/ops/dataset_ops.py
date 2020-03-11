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

def augment_and_resize_images(dataset, img_height, img_width, update_labels = False, pad_resize = True, do_augmentation = True):
    def __augment_and_resize(img, y):
        if do_augmentation:
            _augment_and_clip_image(img, y, update_labels = update_labels)

        return img_ops.resize_image(img, y, img_height, img_width, pad_resize = pad_resize, update_labels = update_labels)

def _augment_and_clip_image(img, y, augments = [img_ops.random_rotation, img_ops.random_brightness_and_contrast, img_ops.random_crop], update_labels = False):
    for aug in augments:
        img, y = img_ops.apply_augment(img, y, aug, update_labels = update_labels)

    img = img_ops.clip_image(img)

    return img, y
    
def get_3_channels(dataset):
    def __get_3_channels(img, y):
        return img_ops.get_3_channels(img, y)
    
    return dataset.map(__get_3_channels, num_parallel_calls=AUTOTUNE)