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

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part
    common_part = create_complex_joint_model(inputs)

    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="complex_multiout_NIH")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)


    return model

def create_rewritten_complex_joint_multioutput(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part
    base_net = rewritten_complex(config)
    common_part = base_net(inputs)
    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="complex_rewritten_multiout_NIH")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)


    return model


def create_bigger_kernel_multioutput(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part

    base_net = bigger_kernel_base(config)
    common_part = base_net(inputs)
    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="bigger_kernel_multiout_NIH")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)


    return model


def create_VGG_multioutput(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    base_net = keras.applications.vgg16.VGG16(include_top=False, weights=None, input_shape=[config.img_height,config.img_width,1], pooling="avg")
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="VGG_multiout_NIH")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)


    return model


def create_densenet_multioutput(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights=None,input_shape=[config.img_height,config.img_width,1], pooling="avg")
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="densenet_multiout_NIH")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)


    return model

def create_resnet_multioutput(config):
    base_resnet = keras.applications.resnet_v2.ResNet50V2(input_shape=[config.img_height,config.img_width,1],pooling="avg",weights=None,)
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    common_part = base_resnet(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    # split into two parts
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease,sex, age],
        name="resnet_multiout_NIH")

    losses = {
    "disease_pred": tfa.losses.focal_loss.SigmoidFocalCrossEntropy(),
    "sex_pred" : "binary_crossentropy",
    "age_pred": "mean_squared_error",
    }
    lossWeights = {"disease_pred": 2.0,"sex_pred" :0.5,"age_pred": 0.005}

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)

    return model


def create_NASnet_multioutupt(config):
    # load base model
    NASnet_model = keras.applications.NASNetMobile(input_shape=[config.img_height,config.img_width,1], include_top=False,weights=None,)
    
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    common_part = NASnet_model(inputs)
    common_part =  tf.keras.layers.GlobalAveragePooling2D()(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)

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

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)

    return model


# This tries a much larger network 22m parameters
def create_NASnet_7x1920_multioutupt(config):
    # load base model
    NASnet_model = NASNet(input_shape=[config.img_height,config.img_width,1], include_top=False,weights=None,
      penultimate_filters=1920,
      num_blocks=7,
      stem_block_filters=96,
      skip_reduction=False,
      filter_multiplier=2,)
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    common_part = NASnet_model(inputs)
    common_part =  tf.keras.layers.GlobalAveragePooling2D()(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    
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

    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics={"disease_pred":"binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"},)

    return model