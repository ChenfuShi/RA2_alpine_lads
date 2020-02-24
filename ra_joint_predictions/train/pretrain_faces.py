
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import datetime
from utils.saver import CustomSaver, _get_tensorboard_callback


def pretrain_faces(model,data_train,data_val,config,model_name,epochs=251):

    # declare custom saver for model
    saver = CustomSaver(model_name,n=25)

    # declare tensorboard
    tensorboard_callback = _get_tensorboard_callback(model_name)
    # fit model for 70 epochs, then we can choose
    H = model.fit(data_train, validation_data=data_val,
    epochs=epochs,steps_per_epoch=2000,validation_steps=10,verbose=2,callbacks=[saver,tensorboard_callback])
