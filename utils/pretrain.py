########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import logging

def pretrain_NIH_chest(model,data_train,data_val,config):
    # function to run training on chest X-ray dataset
    
    # datasets have to be corrected for multioutput
    data_train = data_train.map(_split_dataset_outputs)
    data_val = data_val.map(_split_dataset_outputs)

    saver = CustomSaver()

    H = model.fit(data_train, validation_data=data_val,
    epochs=20,steps_per_epoch=100,validation_steps=5,verbose=2,callbacks=[saver])


def _split_dataset_outputs(x,y):
    # split outputs of dataset for multioutput
    return x,(tf.split(y,[1,16],1)[1],tf.split(y,[1,16],1)[0])


class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        logging.info(logs)
        if epoch // 2 == 0:  # save every 25 epochs
            self.model.save(os.path.join(config.weights_location,"NIH_chest_NASnet_model_{}.hd5".format(epoch)))