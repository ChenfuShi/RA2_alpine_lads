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
    NASnet_model = keras.applications.NASNetMobile(input_shape=[test_config.img_height,test_config.img_width,1], include_top=False,weights=None,)
    
    # create new model with common part
    inputs = keras.layers.Input(shape=[test_config.img_height,test_config.img_width,1])
    common_part = NASnet_model(inputs)
    common_part =  tf.keras.layers.GlobalAveragePooling2D()(common_part)

    # split into two parts
    disease_gender = keras.layers.Dense(16, activation='sigmoid', name='disease_gend_pred')(common_part)

    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease_gender, age],
        name="NASnet_multiout")

    losses = {
    "disease_gend_pred": "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_gend_pred": 1.0, "age_pred": 0.001}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=["binary_accuracy","mae"])

    return model