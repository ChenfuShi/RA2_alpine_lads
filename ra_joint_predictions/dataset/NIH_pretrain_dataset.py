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
from dataset.ops import dataset_ops
import logging

from dataset.base_dataset import base_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class pretrain_dataset_NIH_chest(base_dataset):
    """
    Dataset class for pretrain dataset NIH
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.is_chest = True
        self.cache = config.cache_loc + '/chest/'

    def initialize_pipeline(self, imagenet = False):    
        
        self.data_info = self._get_dataframes(self.config.pretrain_NIH_chest_info)

        self.data_info['Image Index'] = self.data_info["Image Index"].values
        self.data_info[['Image Index','file_type']] = self.data_info['Image Index'].str.split(".",expand=True)
        self.data_info['flip'] = 'N'
        
        x = self.data_info[['Image Index','file_type', 'flip']].values

        data = self.data_info.drop(['file_type', 'flip', 'Image Index'], axis=1).values
        data = data.astype(np.float64)
        
        self.data = data
        
        # get dataset 
        chest_dataset = self._create_dataset(
            x, data, self.config.pretrain_NIH_chest_location, imagenet = imagenet)
        
        # resize the images because it was taking too long
        chest_dataset = dataset_ops.augment_and_resize_images(chest_dataset, 350, 350, pad_resize = True, augments = [])

        # here separate validation set
        chest_dataset, chest_dataset_val = self._create_validation_split(chest_dataset,1000)

        # data processing
        # augmentation happens here
        chest_dataset = self._cache_shuffle_repeat_dataset(chest_dataset, cache = self.cache + 'chest', buffer_size = 5000)
        chest_dataset_val = self._cache_shuffle_repeat_dataset(chest_dataset_val, cache = self.cache + 'chest_val', buffer_size = 1000)
        
        chest_dataset = self._prepare_for_training(chest_dataset, self.config.img_height, self.config.img_width, batch_size = self.config.batch_size, pad_resize = False)
        chest_dataset_val = self._prepare_for_training(chest_dataset_val, self.config.img_height, self.config.img_width, batch_size = self.config.batch_size, augment = False, pad_resize = False)

        return chest_dataset, chest_dataset_val
    
    def _get_dataframes(self, file_csv):
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



