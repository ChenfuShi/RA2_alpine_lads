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


AUTOTUNE = tf.data.experimental.AUTOTUNE

def _prepare_for_training(ds, batch_size=16, cache=True, shuffle_buffer_size=200):
    """ copied from tensorflow tutorial """
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


class train_dataset():
    """
    Dataset class for train and validation split
    """
    def __init__(self,config):
        
        self.config = config

    def initialize_pipeline(self):    
        training_csv_file = os.path.join(self.config.train_location,"training.csv")
        self.data_hands,self.data_feet = self.get_dataframes(training_csv_file)
        # here manage copy over to localscratch

        # here separate validation set

        # get dataset for hands
        hands_dataset = self.init_hands()
        # get dataset for feet
        feet_dataset = self.init_feet()

        # basic dataset processing
        # NEED CACHE LOCATION
        hands_dataset = _prepare_for_training(hands_dataset,self.config.batch_size)
        feet_dataset = _prepare_for_training(feet_dataset,self.config.batch_size)
        # here apply augmentation

        
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

    def copy_over(self):
        # function that takes care of moving dataset to localscratch
        pass

    def augment(self):
        # function that augments datasets
        pass


    # THESE TWO FUNCTIONS LACK THE FLIPPING BASED ON FILENAME!!!!
    def init_feet(self):
        # make feet dataset
        def load_images(file,y):
            path = self.config.train_location+"/"+file+".jpg"
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [ self.config.feet_height,self.config.feet_width])
            return img, y
        dataset = tf.data.Dataset.from_tensor_slices((self.data_feet["Patient_ID"].values, self.data_feet.loc[:, self.data_feet.columns != 'Patient_ID'].values))
        dataset = dataset.map(load_images, num_parallel_calls=AUTOTUNE)
        return dataset

    def init_hands(self):
        # make hands dataset
        def load_images(file,y):
            path = self.config.train_location+"/"+file+".jpg"
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [ self.config.hands_height,self.config.hands_width])
            return img, y
        dataset = tf.data.Dataset.from_tensor_slices((self.data_hands["Patient_ID"].values, self.data_hands.loc[:, self.data_hands.columns != 'Patient_ID'].values))
        dataset = dataset.map(load_images, ) # num_parallel_calls="AUTOTUNE"
        return dataset

