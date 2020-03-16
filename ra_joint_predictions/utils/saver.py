import datetime
import logging
import os

import tensorflow as tf
from tensorflow import keras

def save_pretrained_model(pretrained_model, no_layers_to_remove, model_name):
    if no_layers_to_remove > 0:
        new_model = keras.models.Sequential()
        
        # remove the last x layers, by only adding layers before it to the new model
        idx = -1 * no_layers_to_remove
        for layer in pretrained_model.layers[:idx]:
            new_model.add(layer)

        new_model.save(model_name + '.h5')
    else:
        pretrained_model.save(model_name + '.h5')


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, model_name, n = 25, save_type = "m"):
        super().__init__()
        
        self.model_name = model_name
        self.n = n
        self.save_type = save_type
        
    def on_epoch_end(self, epoch, logs={}):
        #logging.info(logs)
        cur_date = datetime.datetime.now()
        logging.info(f'{cur_date.year}-{cur_date.month}-{cur_date.day}_{cur_date.hour}.{cur_date.minute}.{cur_date.second}')
        
        if epoch % self.n == 0:  # save every n epochs
            if self.save_type == "m":
                keras.models.save_model(self.model,os.path.join('weights', self.model_name + '_model_{}.h5'.format(epoch)))
            else:
                self.model.save_weights(os.path.join('weights', self.model_name + '_model_{}'.format(epoch)))


def _get_tensorboard_callback(model_name,log_dir = 'logs/tensorboard/'):
    log_dir = log_dir + model_name + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)
    
    return tensorboard_callback