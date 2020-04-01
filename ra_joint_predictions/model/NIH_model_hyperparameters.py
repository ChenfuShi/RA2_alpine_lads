########################################




########################################


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
from model.utils.keras_nasnet import NASNet
from model.utils.building_blocks_joints import create_complex_joint_model, bigger_kernel_base, rewritten_complex

def create_complex_joint_multioutput(config):
    # complex model is completely different and should be optimized separately
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part
    common_part = create_complex_joint_model(inputs)

    return _add_common(common_part,"complex_multiout_NIH",inputs)

def create_densenet_multioutput_A(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="max")   
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_A_multiout_NIH",inputs)

def create_densenet_multioutput_B(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="max")   
    # create new model with common part
    common_part = base_net(inputs)


    return _add_common(common_part,"densenet_B_multiout_NIH",inputs)
    
def create_densenet_multioutput_C(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg")   
    # create new model with common part
    common_part = base_net(inputs)


    return _add_common(common_part,"densenet_C_multiout_NIH",inputs)

def create_densenet_multioutput_D(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg")    
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_D_multiout_NIH",inputs)

def create_densenet_multioutput_E(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg")    
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_E_multiout_NIH",inputs)

def create_densenet_multioutput_F(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg")  
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(2048, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(1024, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_F_multiout_NIH",inputs)

def create_densenet_multioutput_G(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg") 
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(2048, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_G_multiout_NIH",inputs)

def create_densenet_multioutput_H(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg") 
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(2048, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_H_multiout_NIH",inputs)

def _add_common(common_part,name,inputs):
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name=name)

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)


    return model
