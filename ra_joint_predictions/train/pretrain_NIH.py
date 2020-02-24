########################################

# DEPRECATED - use module train/pretrain.py instead


########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import datetime
from utils.saver import CustomSaver, _get_tensorboard_callback

def pretrain_NIH_chest(model,data_train,data_val,config,model_name):
    # function to run training on chest X-ray dataset
    
    # datasets have to be corrected for multioutput
    data_train = data_train.map(_split_dataset_outputs)
    data_val = data_val.map(_split_dataset_outputs)

    # declare custom saver for model
    saver = CustomSaver(model_name)

    # declare tensorboard
    log_dir="logs/tensorboard/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # fit model indefinetly
    H = model.fit(data_train, validation_data=data_val,
    epochs=10000,steps_per_epoch=100,validation_steps=10,verbose=2,callbacks=[saver,tensorboard_callback])


def _split_dataset_outputs(x,y):
    # split outputs of dataset for multioutput
    return x,(tf.split(y,[1,1,14],1)[2],tf.split(y,[1,1,14],1)[1],tf.split(y,[1,1,14],1)[0])
