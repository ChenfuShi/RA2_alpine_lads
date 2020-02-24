
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import datetime
from utils.saver import CustomSaver, _get_tensorboard_callback


def pretrain_faces(model,data_train,data_val,config,model_name):
    # function to pretrain to predict faces features
    # should work for feet and hands as well. ok no because we don't have validation split for now
    # declare custom saver for model
    saver = CustomSaver(model_name,n=25)

    # declare tensorboard
    log_dir="logs/tensorboard/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)
    # fit model for 70 epochs, then we can choose
    H = model.fit(data_train, validation_data=data_val,
    epochs=251,steps_per_epoch=2000,validation_steps=10,verbose=2,callbacks=[saver,tensorboard_callback])
