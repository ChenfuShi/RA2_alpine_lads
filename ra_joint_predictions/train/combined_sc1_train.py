import os
import tensorflow as tf
from tensorflow import keras
from utils.config import Config
from utils.saver import CustomSaver, _get_tensorboard_callback
import datetime
import logging
from model.combined_sc1_model import _get_optimizer, _get_adamW

def train_SC1_model(config, model, model_name, train_set, validation_set ,epochs_before=10,epochs_after=51):

    tf.keras.backend.clear_session()

    tensorboard_callback = _get_tensorboard_callback(model_name, log_dir = 'logs/tensorboard_SC1/')

    if epochs_before > 0:
        saver = CustomSaver(model_name + "before", n = 5)
        model.fit(train_set,
            epochs = epochs_before, steps_per_epoch = 40, validation_data = validation_set, validation_steps = 5, verbose = 2, callbacks = [saver, tensorboard_callback])
    
    tf.keras.backend.clear_session()

    for layer in model.layers:
        layer.trainable = True

    adamw_opt = _get_adamW(model, epochs_after, 40)

    # need to recompile after trainable
    model.compile(optimizer = adamw_opt, loss = "mean_absolute_error", metrics = ["mae"])

    
    saver = CustomSaver(model_name + "after", n = 5)
    model.fit(train_set,
        epochs = epochs_after, steps_per_epoch = 40, validation_data = validation_set, validation_steps = 5, verbose = 2, callbacks = [saver, tensorboard_callback])

