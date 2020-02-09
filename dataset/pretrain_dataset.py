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

from dataset.base_dataset import base_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class pretrain_dataset_NIH_chest(base_dataset):
    """
    Dataset class for pretrain dataset NIH
    """

    def __init__(self, config):
        super().__init__(config)

    def initialize_pipeline(self):    
        
        self.data_info = _get_dataframes(self.config.pretrain_NIH_chest_info)

        # get dataset 
        chest_dataset = _init_dataset(self.data_info,self.config.pretrain_NIH_chest_location)
        
        # here separate validation set
        chest_dataset, chest_dataset_val = super()._create_validation_split(chest_dataset)

        # data processing
        # augmentation happens here

        chest_dataset = super()._prepare_for_training(chest_dataset, self.config.img_width, self.config.img_height, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'chest')
        chest_dataset = super()._prepare_for_training(chest_dataset, self.config.img_width, self.config.img_height, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'chest_val')

        return chest_dataset, chest_dataset_val
    
def _get_dataframes(file_csv):
    pretrain_NIH_info = pd.read_csv(file_csv)
    pretrain_NIH_info['Finding Labels'] = pretrain_NIH_info['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    pretrain_NIH_info['Finding Labels'] = pretrain_NIH_info['Finding Labels'].map(lambda x: x.split('|'))
    all_labels = set(pretrain_NIH_info['Finding Labels'].sum())
    all_labels = [x for x in all_labels if len(x)>0]   
    for c_label in all_labels:
        pretrain_NIH_info[c_label] = pretrain_NIH_info['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    pretrain_NIH_info["Gender_M"] = pretrain_NIH_info['Patient Gender'].map(lambda gen: 1.0 if "M" in gen else 0)
    useful_cols=["Image Index","Patient Age","Gender_M"] + all_labels

    return pretrain_NIH_info[useful_cols]

def _init_dataset(df_data, pretrain_location):
    def __load_image(file, y):
        file_path = pretrain_location + "/" + file 
        
        img = tf.io.read_file(file_path)    
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img, y

    dataset = tf.data.Dataset.from_tensor_slices((df_data["Image Index"].values, df_data.loc[:, df_data.columns != 'Image Index'].values))

    dataset = dataset.map(__load_image, num_parallel_calls=AUTOTUNE)
        
    return dataset

