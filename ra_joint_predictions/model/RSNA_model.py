import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import model.NIH_model as NIH
from model.utils.building_blocks_joints import _fc_block
from model.utils.building_blocks_joints import create_complex_joint_model


def complex_joint_finetune_model(config, no_joint_types = 10, weights = "weights/NIH_new_pretrain_model_100.h5"):
    NIH_model = keras.models.load_model(weights)

    NEW_model = keras.Model(NIH_model.input, NIH_model.layers[-4].output)

    for layer in NEW_model.layers:
        layer.trainable=False

    # split into three parts

    boneage_fc = _fc_block(NEW_model.output, 32, 'boneage_1')
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(boneage_fc)

    sex_fc = _fc_block(NEW_model.output, 32, 'sex_1')
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(sex_fc)

    joint_type_fc = _fc_block(NEW_model.output, 64, 'joint_type_1')
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred')(joint_type_fc)

    # get final model
    model = keras.models.Model(
        inputs=NEW_model.input,
        outputs=[boneage, sex, joint_type],
        name='rsna_complex_multiout')

    losses = {
        'boneage_pred': 'mean_squared_error',
        'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': 'categorical_crossentropy',
    }

    lossWeights = {'boneage_pred': 0.005, 'sex_pred': 2, 'joint_type_pred': 1}

    model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'categorical_accuracy'})

    return model


def create_rsna_NASnet_multioutupt(config, no_joint_types = 13):
    inputs = keras.layers.Input(shape=[config.joint_img_height, config.joint_img_width, 1])
    
    base_model = create_complex_joint_model(inputs)

    # split into three parts

    boneage_fc = _fc_block(base_model, 32, 'boneage_1')
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(boneage_fc)

    sex_fc = _fc_block(base_model, 32, 'sex_1')
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(sex_fc)

    joint_type_fc = _fc_block(base_model, 64, 'joint_type_1')
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred')(joint_type_fc)

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
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'categorical_accuracy'})

    return model



