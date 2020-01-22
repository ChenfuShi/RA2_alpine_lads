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

AUTOTUNE = tf.data.experimental.AUTOTUNE

class train_dataset():
    """
    Dataset class for train and validation split
    """

    def __init__(self,config):
        self.config = config

        self.augments = [aug.random_brightness_and_contrast, aug.random_crop, aug.random_rotation]

    def initialize_pipeline(self):    
        training_csv_file = os.path.join(self.config.train_location,"training.csv")
        self.data_hands,self.data_feet = self.get_dataframes(training_csv_file)

        # get dataset for hands
        hands_dataset = self.init_dataset(self.data_hands,"RF")
        # get dataset for feet
        feet_dataset = self.init_dataset(self.data_feet,"RH")

        # here separate validation set
        if self.config.have_val:
            # shuffle to get random split every time. Be careful about this!
            # hands_dataset = hands_dataset.shuffle()
            # feet_dataset = feet_dataset.shuffle()
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

    def get_dataframes(self,training_csv):
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
        return data_hands,data_feet

    def init_dataset(self,df_data,flip_str):
        def load_images(file,y):
            path = self.config.train_location+"/"+file+".jpg"
            img = tf.io.read_file(path)
            
            img_or = tfio.experimental.image.decode_jpeg_exif(img)
            img = tfa.image.rotate(img, img_or)
            
            img = tf.image.decode_jepg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            if flip_str in str(file):
                img = tf.image.flip_left_right(img)
            return img, y
        dataset = tf.data.Dataset.from_tensor_slices((df_data["Patient_ID"].values, df_data.loc[:, df_data.columns != 'Patient_ID'].values))
        dataset = dataset.map(load_images, num_parallel_calls=AUTOTUNE)
        return dataset

    def _prepare_for_training(self,ds, augment,cache=True, shuffle_buffer_size=200):            
        def resize_img(img,y):
            # function that just resizes the image
            img = tf.image.resize(img, [ self.config.img_height,self.config.img_width])
            return img, y

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
            dataset = _randomly_augment_dataset(dataset)

        # Resize image after random augmentation
        ds = ds.map(resize_img, num_parallel_calls=AUTOTUNE)

        # batch
        ds = ds.batch(self.config.batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def _randomly_augment_dataset(dataset):
        for aug in self.augments:
                dataset = _apply_random_augment(dataset, aug)

        # After augmentations, scale values back to lie between 0 & 1
        dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=AUTOTUNE)

        return dataset

    def _apply_random_augment(dataset, aug, cutoff = 0.75):
        # Randomly apply each augmentation 1 - cutoff% of the time
        return dataset.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > cutoff, lambda: aug(x), lambda: x), num_parallel_calls=AUTOTUNE)