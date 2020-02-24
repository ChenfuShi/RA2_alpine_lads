import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

from utils.losses import categorical_focal_loss
from model import create_bigger_kernel_base, create_complex_joint_model

def create_rsna_NASnet_multioutupt(img_height, img_width, no_joints_types = 13):
    inputs = keras.layers.Input(shape=[img_height, img_width, 1])
    
    base_model = create_complex_joint_model(inputs)

    # split into three parts
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(base_model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(base_model)
    joint_type = keras.layers.Dense(no_joints_types, activation = 'softmax', name = 'joint_type_pred')(base_model)

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
