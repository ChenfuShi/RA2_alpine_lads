########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa


def create_NASnet_multioutupt(config):
    # load base model
    NASnet_model = keras.applications.NASNetMobile(input_shape=[config.img_height,config.img_width,1], include_top=False,weights=None,)
    
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    common_part = NASnet_model(inputs)
    common_part =  tf.keras.layers.GlobalAveragePooling2D()(common_part)

    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="NASnet_multiout")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    #cl_weights = { 'disease_pred': {0: 1 , 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1}}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)
    #class_weight = cl_weights)

    return model