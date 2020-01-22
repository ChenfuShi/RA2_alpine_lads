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

import augments as aug

from augmentation import image_augmentor as augmentor, augments as augs

AUTOTUNE = tf.data.experimental.AUTOTUNE

class train_dataset():
    """
    Dataset class for train and validation split
    """

    def __init__(self,config):
        self.config = config

        # Init augments that should be applied to the dataset
        self.augments = [augs.random_brightness_and_contrast, augs.random_crop, augs.random_rotation]

    def initialize_pipeline(self):    
        training_csv_file = os.path.join(self.config.train_location,"training.csv")
        
        self.data_hands, self.data_feet = _get_dataframes(training_csv_file)

        # get dataset for hands
        hands_dataset = _init_dataset(self.data_hands, self.config.train_location, "RF")
        # get dataset for feet
        feet_dataset = _init_dataset(self.data_feet, self.config.train_location, "RH")

        # here separate validation set
        if self.config.have_val:
            hands_dataset_val = hands_dataset.take(50) 
            hands_dataset = hands_dataset.skip(50)
            feet_dataset_val = feet_dataset.take(50) 
            feet_dataset = feet_dataset.skip(50)
        
        # data processing
        # augmentation happens here
        hands_dataset = self._prepare_for_training(hands_dataset,self.config.augment,self.config.cache_loc + "hands")
        feet_dataset = self._prepare_for_training(feet_dataset,self.config.augment,self.config.cache_loc + "feet")

        if self.config.have_val:
            hands_dataset_val = self._prepare_for_training(hands_dataset_val,False)
            feet_dataset_val = self._prepare_for_training(feet_dataset_val,False)

        if self.config.have_val:
            return hands_dataset,feet_dataset,hands_dataset_val,feet_dataset_val
        else:
            return hands_dataset,feet_dataset

    def _prepare_for_training(self,ds, augment,cache=True, shuffle_buffer_size=200):            
        if cache:
            if isinstance(cache, str):
                try:
                    os.makedirs(os.path.expanduser(cache),exist_ok=True)
                    ds = ds.cache(os.path.expanduser(cache))
                except FileNotFoundError:
                    # this means that we are not on a CSF node and we are not allowed to make this folder
                    pass
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        if augment:
            dataset = augmentor.randomly_augment_dataset(dataset, self.augments)

        dataset = _resize_img(dataset, self.config.img_height, self.config.img_width)

        # batch
        ds = ds.batch(self.config.batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds
    
def _get_dataframes(training_csv):
    info = pd.read_csv(training_csv)
    features = info.columns
    parts = ["LH","RH","LF","RF"]
    dataframes = {}
    for part in parts:
        dataframes[part] = info.loc[:,["Patient_ID"]+[s for s in features if part in s]].copy()
        dataframes[part]["total_fig_score"] = dataframes[part].loc[:,[s for s in features if part in s]].sum(axis=1)
        dataframes[part]["Patient_ID"] = dataframes[part]["Patient_ID"].astype(str) + f"-{part}"
    
    # use left as reference
    # flip the Rights
    dataframes["RH"].columns = dataframes["LH"].columns 
    dataframes["RF"].columns = dataframes["LF"].columns 
        
    data_hands = pd.concat((dataframes["RH"],dataframes["LH"]))
    data_feet = pd.concat((dataframes["RF"],dataframes["LF"]))
        
    return data_hands, data_feet

def _init_dataset(df_data, train_location, flip_str):
    dataset = tf.data.Dataset.from_tensor_slices((df_data["Patient_ID"].values, df_data.loc[:, df_data.columns != 'Patient_ID'].values))

    dataset = _load_images(dataset, train_location, flip_str)
        
    return dataset

def _load_images(dataset, train_location, flip_str):
    def __load(file, y):
        file_path = train_location + "/fixed/" + file + ".jpg"
    
        img = tf.io.read_file(file_path)
            
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
            
        if flip_str in str(file):
            img = tf.image.flip_left_right(img)
                
        return img, y

    return dataset.map(__load, num_parallel_calls=AUTOTUNE)

def _resize_img(dataset, img_height, img_width):
    def __resize(img, y):
        img = tf.image.resize(img, [ img_height, img_width])

        return img, y

    return dataset.map(__resize, num_parallel_calls=AUTOTUNE)