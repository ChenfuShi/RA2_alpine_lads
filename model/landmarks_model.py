########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import model.keras_nasnet


def landmarks_model_pretrain(config):
    # model for pretraining on faces

    model = _landmarks_base(config)
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation='linear'))

    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

    return model

def landmarks_model_feet(config,weights=None):
    # model for feet
    # weights are only the pretrain weights. if you have normal weights please load them afterwards
    if weights == None:
        model = _landmarks_base(config)
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(12, activation='linear'))

        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])

    else:
        pretrained_model = landmarks_model_pretrain(config)
        pretrained_model.load_weights(weights)
        model=keras.models.Sequential()
        # remove three layers from pretrain model
        for layer in pretrained_model.layers[:-3]:
            model.add(layer)
        # add the layers back
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(12, activation='linear'))
        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    return model

def landmarks_model_hands(config,weights=None):
    pass

def _landmarks_base(config):

    # model = keras.models.Sequential([
    #     keras.layers.Conv2D(filters = 16, kernel_size=(3,3),activation="relu",input_shape=[config.landmarks_img_height,config.landmarks_img_width,1]),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters = 16, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Flatten(),

    # ])

    # try something bigger
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu",input_shape=[config.landmarks_img_height,config.landmarks_img_width,1]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 256, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 256, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),

    ])


    return model