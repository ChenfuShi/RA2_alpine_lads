########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf


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
    # model for hands
    # weights are only the pretrain weights. if you have normal weights please load them afterwards
    if weights == None:
        model = _landmarks_base(config)
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(26, activation='linear'))

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
        model.add(keras.layers.Dense(26, activation='linear'))
        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    return model

def _landmarks_base(config):
    # original, the one i don't know how many epochs i did
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

    # try something bigger (medium)
    # model = keras.models.Sequential([
    #     keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu",input_shape=[config.landmarks_img_height,config.landmarks_img_width,1]),
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
    #     keras.layers.Conv2D(filters = 256, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 256, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Flatten(),

    # ])

    # try something much shallower
    # model = keras.models.Sequential([
    #     keras.layers.Conv2D(filters = 16, kernel_size=(3,3),activation="relu",input_shape=[config.landmarks_img_height,config.landmarks_img_width,1]),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Flatten(),
    # ])

    # last try with bigger kernels 
    # ok this one seems best but i've changed it to include more filters and train for much longer
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu",input_shape=[config.landmarks_img_height,config.landmarks_img_width,1]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),

    ])

    return model



def resnet_landmarks_model(config,n_outputs=10,weights=None,outputs_before=10):
    base_resnet = keras.applications.resnet_v2.ResNet50V2(input_shape=[config.landmarks_img_height,config.landmarks_img_width,1],pooling="avg",weights=None,)
    model = keras.models.Sequential([base_resnet])

    if weights == None:
        model.add(keras.layers.Dense(n_outputs, activation='linear'))

        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    else:
        pretrained_model = resnet_landmarks_model(config,outputs_before)
        pretrained_model.load_weights(weights)
        model=keras.models.Sequential()
        # remove 1 layers from pretrain model
        for layer in pretrained_model.layers[:-1]:
            model.add(layer)
        # add the layers back
        model.add(keras.layers.Dense(n_outputs, activation='linear'))
        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    return model


def nasnet_landmarks_model(config,n_outputs=10,weights=None,outputs_before=10):
    base_net = keras.applications.NASNetMobile(input_shape=[config.landmarks_img_height,config.landmarks_img_width,1],weights=None,pooling="avg",)
    model = keras.models.Sequential([base_net])

    if weights == None:
        model.add(keras.layers.Dense(n_outputs, activation='linear'))

        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    else:
        pretrained_model = nasnet_landmarks_model(config,outputs_before)
        pretrained_model.load_weights(weights)
        model=keras.models.Sequential()
        # remove 1 layers from pretrain model
        for layer in pretrained_model.layers[:-1]:
            model.add(layer)
        # add the layers back
        model.add(keras.layers.Dense(n_outputs, activation='linear'))
        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    return model