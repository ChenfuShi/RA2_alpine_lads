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
import dataset.ops.dataset_ops as ds_ops
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
        
        self.data_info = pd.read_csv(self.config.landmarks_faces_info,sep="\s+|\t+|\s+\t+|\t+\s+",skiprows=1,engine="python")

        self.data_info['idx'] = self.data_info.index.values
        self.data_info[['idx','file_type']] = self.data_info['idx'].str.split(".",expand=True)
        self.data_info['flip'] = 'N'
        
        x = self.data_info[['idx','file_type', 'flip']].values

        data = self.data_info[self.data_info.columns.difference(['file_type', 'flip','idx'])].values
        data = data.astype(np.float64)
        # get dataset 

        dataset = self._create_dataset(
            x, data, self.config.landmarks_faces_location, update_labels=True)
    
        # here separate validation set
        dataset, dataset_val = super()._create_validation_split(dataset,5000)

        # data processing
        # augmentation happens here
        dataset = super()._prepare_for_training(dataset, self.config.landmarks_img_width, self.config.landmarks_img_height, 
            batch_size = self.config.batch_size, cache = self.config.cache_loc + 'faces',update_labels=True)
        dataset_val =super()._prepare_for_training(dataset_val, self.config.landmarks_img_width, self.config.landmarks_img_height, 
            batch_size = self.config.batch_size, cache = self.config.cache_loc + 'faces_val',update_labels=True)
        
        return dataset, dataset_val
