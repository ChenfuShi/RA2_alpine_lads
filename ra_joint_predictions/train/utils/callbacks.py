import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

class AdamWWarmRestartCallback(keras.callbacks.Callback):
    def __init__(self, restart_epochs = 50):
        super().__init__()
        
        self.restart_epochs = restart_epochs
    
    def on_epoch_begin(self, epoch, logs = None):
        if epoch != 0 & epoch % self.restart_epochs == 0:
            K.set_value(self.model.optimizer.t_cur, 0)