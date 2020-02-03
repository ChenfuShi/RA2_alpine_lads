########################################




########################################


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils.config import Config
from PIL import Image
from tensorflow import keras

import logging
import dataset.dataset_ops as ops

AUTOTUNE = tf.data.experimental.AUTOTUNE


class landmark_pretrain_faces_dataset():
    """
    Dataset class for pretrain dataset NIH
    """

    def __init__(self,config):
        self.config = config

    def initialize_pipeline(self):    
        
        self.data_info =  pd.read_csv(landmark_info,sep="\s+|\t+|\s+\t+|\t+\s+",skiprows=1)

        # get dataset 
        faces = _init_dataset(self.data_info,self.config.landmarks_faces_location)

        # here separate validation set
        dataset_val = dataset.take(5000) 
        dataset = dataset.skip(5000)

        # data processing
        # augmentation happens here
        dataset = self._prepare_for_training(dataset,self.config.augment,self.config.cache_loc + "faces")
        dataset_val = self._prepare_for_training(dataset_val,False,self.config.cache_loc + "faces_val")
        return dataset, dataset_val



    def _prepare_for_training(self, ds, augment, cache=True, shuffle_buffer_size=200):            
        if cache:
            if isinstance(cache, str):
                try:
                    os.makedirs(os.path.expanduser(cache),exist_ok=True)
                    ds = ds.cache(os.path.expanduser(cache))
                except FileNotFoundError:
                    logging.warn("Missing permissions to create directory for caching!")

                    pass                                                                                            # Missing permission to create cache folder
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)                                                            # Shuffle dataset
        ds = ds.repeat()                                                                                            # Repeat dataset entries

        ## this bit change with stuff needed for landmarks scaling
        # if augment:
        #     ds = _augment_images(ds)

        # ds = _resize_images(ds, self.config.landmarks_img_width, self.config.landmarks_img_height)

        ds = ds.batch(self.config.batch_size)                                                                       # Enable batching
        ds = ds.prefetch(buffer_size=AUTOTUNE)                                                                      # Fetch batches in background while model is training

        return ds
    

def _init_dataset(df_data, pretrain_location):
    def __load_image(file, y):
        file_path = self.config.landmarks_faces_location + file 

        img = tf.io.read_file(file_path)    
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img, y

    dataset =  tf.data.Dataset.from_tensor_slices((self.data_info.index.values, self.data_info.values))

    dataset = dataset.map(__load_image, num_parallel_calls=AUTOTUNE)
        
    return dataset

def _augment_images(ds):
    return ops.randomly_augment_images(ds)

def _resize_images(ds, img_width, img_height):
    return ops.resize_images(ds, img_width, img_height)
