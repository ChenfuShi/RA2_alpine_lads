import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import datetime
from utils.saver import CustomSaver, _get_tensorboard_callback

def pretrain_NIH_chest(model, data_train, data_val, config, model_name, epochs = 251):
    # function to run training on chest X-ray dataset
    
    # datasets have to be corrected for multioutput
    data_train = data_train.map(_split_dataset_outputs)
    data_val = data_val.map(_split_dataset_outputs)

    # declare custom saver for model
    saver = CustomSaver(model_name,n=25)

    # declare tensorboard
    # tensorboard_callback = _get_tensorboard_callback(model_name)  ## tensorboard_callback
    # fit model indefinetly
    H = model.fit(data_train, validation_data=data_val,
    epochs=epochs,steps_per_epoch=1000,validation_steps=15,verbose=2,callbacks=[saver])


def _split_dataset_outputs(x,y):
    # split outputs of dataset for multioutput
    # return x,(tf.split(y,[1,1,14],1)[2],tf.split(y,[1,1,14],1)[1],tf.split(y,[1,1,14],1)[0])
    
    return x,(tf.split(y,[1,1,14],1)[2], tf.split(y,[1,1,14],1)[0])

