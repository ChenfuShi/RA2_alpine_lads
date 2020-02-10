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

from dataset.base_dataset import base_dataset
import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE


class landmark_pretrain_faces_dataset(base_dataset):
    """
    Dataset class for pretrain dataset faces celeb_a
    """
    def __init__(self, config):
        super().__init__(config)

    def create_train_dataset(self):    
        
        self.data_info =  pd.read_csv(self.config.landmarks_faces_info,sep="\s+|\t+|\s+\t+|\t+\s+",skiprows=1,engine="python")

        # get dataset 
        faces = self._init_dataset(self.data_info,self.config.landmarks_faces_location)

        # here separate validation set
        dataset, dataset_val = super()._create_validation_split(faces,5000)

        # data processing
        # augmentation happens here
        dataset = super()._prepare_for_training(dataset, self.config.landmarks_img_width, self.config.landmarks_img_height, 
            batch_size = self.config.batch_size, cache = self.config.cache_loc + 'faces',update_labels=True)
        dataset_val =super()._prepare_for_training(dataset_val, self.config.landmarks_img_width, self.config.landmarks_img_height, 
            batch_size = self.config.batch_size, cache = self.config.cache_loc + 'faces_val',update_labels=True)
        
        return dataset, dataset_val

    def _init_dataset(self,df_data, pretrain_location):
        def __load_image(file, y):
            file_path = self.config.landmarks_faces_location + file 

            img = tf.io.read_file(file_path)    
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            y = tf.cast(y,dtype=tf.float64)
            return img, y

        dataset =  tf.data.Dataset.from_tensor_slices((self.data_info.index.values, self.data_info.values))

        dataset = dataset.map(__load_image, num_parallel_calls=AUTOTUNE)
            
        return dataset

