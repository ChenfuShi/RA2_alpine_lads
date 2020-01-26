########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import datetime

def pretrain_NIH_chest(model,data_train,data_val,config,model_name):
    # function to run training on chest X-ray dataset
    
    # datasets have to be corrected for multioutput
    data_train = data_train.map(_split_dataset_outputs)
    data_val = data_val.map(_split_dataset_outputs)

    # declare custom saver for model
    saver = CustomSaver(model_name)

    # declare tensorboard
    log_dir="logs/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # fit model indefinetly
    H = model.fit(data_train, validation_data=data_val,
    epochs=10000,steps_per_epoch=100,validation_steps=10,verbose=2,callbacks=[saver,tensorboard_callback])


def _split_dataset_outputs(x,y):
    # split outputs of dataset for multioutput
    return x,(tf.split(y,[1,1,14],1)[2],tf.split(y,[1,1,14],1)[1],tf.split(y,[1,1,14],1)[0])


class CustomSaver(keras.callbacks.Callback):
    def __init__(self,model_name):
        self.model_name = model_name
        super().__init__()
        
    def on_epoch_end(self, epoch, logs={}):
        logging.info(logs)
        cur_date = datetime.datetime.now()
        logging.info(f"{cur_date.year}-{cur_date.month}-{cur_date.day}_{cur_date.hour}.{cur_date.minute}.{cur_date.second}")
        if epoch % 25 == 0:  # save every 25 epochs
            self.model.save_weights(os.path.join("weights",self.model_name + "_model_{}".format(epoch)))