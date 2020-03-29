import os
import tensorflow as tf
from tensorflow import keras
from utils.config import Config
from utils.saver import CustomSaver, _get_tensorboard_callback
import datetime
import logging

def train_SC1_model(config, model, model_name, train_set, validation_set ,epochs_before=10,epochs_after=51):

    tensorboard_callback = _get_tensorboard_callback(model_name, log_dir = 'logs/tensorboard_SC1/')

    if epochs_before > 0:
        saver = CustomSaver(model_name + "before", n = 5)
        model.fit_generator(train_set,
            epochs = epochs_before, steps_per_epoch = 20, validation_data = validation_set, validation_steps = 5, verbose = 2, callbacks = [saver, tensorboard_callback])

    for layer in model.layers:
        layer.trainable = True

    # need to recompile after trainable
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss = "mean_squared_error", metrics = ["mae"])


    saver = CustomSaver(model_name + "after", n = 5)
    model.fit_generator(train_set,
        epochs = epochs_after, steps_per_epoch = 20, validation_data = validation_set, validation_steps = 5, verbose = 2, callbacks = [saver, tensorboard_callback])

