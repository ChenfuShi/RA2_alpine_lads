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

from dataset.base_dataset import base_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class pretrain_dataset_NIH_chest(base_dataset):
    """
    Dataset class for pretrain dataset NIH
    """

    def __init__(self, config):
        super().__init__(config)

    def initialize_pipeline(self):    
        
        self.data_info = self._get_dataframes(self.config.pretrain_NIH_chest_info)

        self.data_info['Image Index'] = self.data_info["Image Index"].values
        self.data_info[['Image Index','file_type']] = self.data_info['Image Index'].str.split(".",expand=True)
        self.data_info['flip'] = 'N'
        
        x = self.data_info[['Image Index','file_type', 'flip']].values

        data = self.data_info.drop(['file_type', 'flip', 'Image Index'], axis=1).values
        data = data.astype(np.float64)

        # get dataset 
        chest_dataset = self._create_dataset(
            x, data, self.config.pretrain_NIH_chest_location)
        
        # here separate validation set
        chest_dataset, chest_dataset_val = super()._create_validation_split(chest_dataset,1000)

        # data processing
        # augmentation happens here

        chest_dataset = super()._prepare_for_training(chest_dataset, self.config.img_height, self.config.img_width, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'chest')
        chest_dataset_val = super()._prepare_for_training(chest_dataset_val, self.config.img_height, self.config.img_width, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'chest_val')

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



