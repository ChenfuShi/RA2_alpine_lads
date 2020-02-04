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

from dataset.base_dataset import base_dataset

class train_dataset(base_dataset):
    def __init__(self, config):
        super().__init__(config)

    def create_train_dataset(self):
        training_csv_file = os.path.join(self.config.train_location, "training.csv")

        self.data_hands, self.data_feet = _get_dataframes(training_csv_file)

        data_dir = os.path.join(self.config.train_location, self.config.fixed_directory)

        hands_dataset = super()._create_dataset(self.data_hands, data_dir)
        feet_dataset = super()._create_dataset(self.data_feet, data_dir)
        
        if self.config.have_val:
            hands_dataset, hands_dataset_val = super()._create_validation_split(hands_dataset)
            feet_dataset, feet_dataset_val = super()._create_validation_split(feet_dataset)

        hands_dataset = super()._prepare_for_training(hands_dataset, self.config.img_width, self.config.img_height, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'hands')
        feet_dataset = super()._prepare_for_training(feet_dataset, self.config.img_width, self.config.img_height, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'hands')

        if self.config.have_val:
            hands_dataset_val = super()._prepare_for_training(hands_dataset_val, self.config.img_width, self.config.img_height, batch_size = self.config.batch_size, augment = False)
            feet_dataset_val = super()._prepare_for_training(feet_dataset_val, self.config.img_width, self.config.img_height, batch_size = self.config.batch_size, augment = False)

            return hands_dataset, feet_dataset, hands_dataset_val, feet_dataset_val
        else:
            return hands_dataset, feet_dataset
    
def _get_dataframes(training_csv):
    info = pd.read_csv(training_csv)
    features = info.columns
    parts = ["LH","RH","LF","RF"]
    dataframes = {}
    for part in parts:
        flip = 'N'
        if(part.startswith('R')):
            flip = 'Y'
        
        dataframes[part] = info.loc[:,["Patient_ID"]+[s for s in features if part in s]].copy()
        dataframes[part]["total_fig_score"] = dataframes[part].loc[:,[s for s in features if part in s]].sum(axis=1)
        dataframes[part]["Patient_ID"] = dataframes[part]["Patient_ID"].astype(str) + f"-{part}"
        dataframes[part]["flip"] = flip
    
    # use left as reference
    # flip the Rights
    dataframes["RH"].columns = dataframes["LH"].columns 
    dataframes["RF"].columns = dataframes["LF"].columns 
        
    data_hands = pd.concat((dataframes["RH"],dataframes["LH"]))
    data_feet = pd.concat((dataframes["RF"],dataframes["LF"]))
        
    return data_hands, data_feet
