import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

from utils.losses import categorical_focal_loss
from model import create_bigger_kernel_base

def create_rsna_NASnet_multioutupt(img_height, img_width, no_joints_types = 13):
    inputs = keras.layers.Input(shape=[img_height, img_width, 1])
    
    base_model = create_bigger_kernel_base(inputs)

    dense_layer_1 = Dense(512, activation = 'relu')(base_model)
    dense_layer_2 = Dense(256, activation = 'relu')(dense_layer_1)
    dense_layer_3 = Dense(128, activation = 'relu')(dense_layer_2)
    dense_layer_4 = Dropout(0.25)(dense_layer_3)

    # split into three parts
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(dense_layer_4)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(dense_layer_4)
    joint_type = keras.layers.Dense(no_joints_types, activation = 'softmax', name = 'joint_type_pred')(dense_layer_4)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[boneage, sex, joint_type],
        name='rsna_NASnet_multiout')

    losses = {
        'boneage_pred': 'mean_squared_error',
        'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': 'categorical_crossentropy',
    }

    lossWeights = {'boneage_pred': 0.005, 'sex_pred': 2, 'joint_type_pred': 1}

    model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'binary_accuracy'})

    return model
