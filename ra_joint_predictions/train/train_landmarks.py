
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import datetime
from utils.saver import CustomSaver, _get_tensorboard_callback


def train_landmarks(model,data_train,data_val,config,model_name):

    saver = CustomSaver(model_name,n=50)

    # declare tensorboard
    log_dir="logs/tensorboard/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)
    # fit model for 500 epochs, then we can choose
    H = model.fit(data_train, validation_data=data_val,
    epochs=2501,steps_per_epoch=20,verbose=2,validation_steps=5,callbacks=[saver,tensorboard_callback])

