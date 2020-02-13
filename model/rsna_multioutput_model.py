import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

from utils.losses import categorical_focal_loss

def create_rsna_NASnet_multioutupt(img_height, img_width, no_joints_types = 13):
    # load base model
    NASnet_model = keras.applications.NASNetMobile(input_shape=[img_height, img_width, 1], include_top = False, weights = None)
    
    # create new model with common part
    inputs = keras.layers.Input(shape = [img_height, img_width, 1])
    common_part = NASnet_model(inputs)
    common_part =  tf.keras.layers.GlobalAveragePooling2D()(common_part)

    # split into two parts
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(common_part)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(common_part)
    joint_type = keras.layers.Dense(no_joints_types, activation = 'softmax', name = 'joint_type_pred')(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[boneage, sex, joint_type],
        name='rsna_NASnet_multiout')

    losses = {
        'boneage_pred': 'mean_squared_error',
        'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': categorical_focal_loss(),
    }
    lossWeights = {'boneage_pred': 0.005, 'sex_pred': 0.5, 'joint_type_pred': 2}

    model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'binary_accuracy'})

    return model
