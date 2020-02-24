import datetime
import logging
import os

from tensorflow import keras

class CustomSaver(keras.callbacks.Callback):
    def __init__(self, model_name, n = 25):
        super().__init__()
        
        self.model_name = model_name
        self.n = n
        
    def on_epoch_end(self, epoch, logs={}):
        logging.info(logs)
        cur_date = datetime.datetime.now()
        logging.info(f'{cur_date.year}-{cur_date.month}-{cur_date.day}_{cur_date.hour}.{cur_date.minute}.{cur_date.second}')
        
        if epoch % self.n == 0:  # save every n epochs
            self.model.save_weights(os.path.join('weights', self.model_name + '_model_{}'.format(epoch)))