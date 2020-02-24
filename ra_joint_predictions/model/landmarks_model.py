########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
from model.utils.building_blocks import bigger_kernel_base

def basic_landmarks_model(config,n_outputs=10,weights=None,outputs_before=10):
    base_original_model = bigger_kernel_base(config)
    
    if weights == None:
        model = keras.models.Sequential([base_original_model])
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n_outputs, activation='linear'))

        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    else:
        pretrained_model = basic_landmarks_model(config,outputs_before)
        pretrained_model.load_weights(weights)
        model=keras.models.Sequential()
        # remove 1 layers from pretrain model 
        # note this does not reset the 256 dense layer
        for layer in pretrained_model.layers[:-1]:
            model.add(layer)
        # add the layers back
        model.add(keras.layers.Dense(n_outputs, activation='linear'))
        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])

    return model


def resnet_landmarks_model(config,n_outputs=10,weights=None,outputs_before=10):
    base_resnet = keras.applications.resnet_v2.ResNet50V2(input_shape=[config.landmarks_img_height,config.landmarks_img_width,1],pooling="avg",weights=None,)
    
    if weights == None:
        model = keras.models.Sequential([base_resnet])
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
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

    if weights == None:
        model = keras.models.Sequential([base_net])
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
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


def densenet_landmarks_model(config,n_outputs=10,weights=None,outputs_before=10):
    base_densenet = keras.applications.densenet.DenseNet121(input_shape=[config.landmarks_img_height,config.landmarks_img_width,1],pooling="avg",weights=None,)
    
    if weights == None:
        model = keras.models.Sequential([base_densenet])
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n_outputs, activation='linear'))

        model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])
    else:
        pretrained_model = densenet_landmarks_model(config,outputs_before)
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













## keeping this for backup and backwards compatibility

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