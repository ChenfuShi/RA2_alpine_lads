import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os
import tensorflow as tf
import tensorflow_addons as tfa
from model.utils.keras_nasnet import NASNet
from model.utils.building_blocks_joints import create_complex_joint_model, bigger_kernel_base, rewritten_complex, rewritten_elu, relu_joint_res_net, complex_rewritten, small_densenet, bottlenecked_small_dense, small_resnet_with_bottleneck, small_vgg_with_bottleneck

def create_rewritten_elu(config):
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part
    base_net = rewritten_elu(config)
    common_part = base_net(inputs)

    return _add_common(common_part,"rewritten_elu_NIH",inputs)

def create_small_bottlenecked_vgg(config, model_name, inputs = None):
    if inputs is None:
        inputs = keras.layers.Input(shape = [config.chest_img_height, config.chest_img_width, 1])

    model = small_vgg_with_bottleneck(inputs)
    
    optimizer = keras.optimizers.Adam(learning_rate = 3e-4)
    
    model = _add_common(model, model_name, inputs, optimizer = optimizer)
    
    return model

def create_small_dense(config, model_name, inputs = None):
    if inputs is None:
        inputs = keras.layers.Input(shape = [config.chest_img_height, config.chest_img_width, 1])

    model = bottlenecked_small_dense(inputs)
    
    optimizer = keras.optimizers.Adam(learning_rate = 3e-4)
    
    model = _add_common(model, model_name, inputs, optimizer = optimizer)
    
    return model

def create_small_resnet(config):
    inputs = keras.layers.Input(shape = [config.img_height, config.img_width, 1])
    
    model = small_resnet_with_bottleneck(inputs)
    
    optimizer = keras.optimizers.Adam(learning_rate = 3e-4)
    
    model = _add_common(model, 'small_bottlenecked_res_900k_NIH_no_sex', inputs, optimizer = optimizer)
    
    return model

def create_complex_joint_multioutput(config):
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part
    common_part = create_complex_joint_model(inputs)

    return _add_common(common_part,"complex_multiout_NIH",inputs)


def create_rewritten_complex_joint_multioutput(config):
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part
    base_net = rewritten_complex(config, decay = None, use_dense = False)
    common_part = base_net(inputs)

    return _add_common(common_part,"complex_rewritten_gap_NIH",inputs)


def create_bigger_kernel_multioutput(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])

    # create new model with common part

    base_net = bigger_kernel_base(config)
    common_part = base_net(inputs)

    return _add_common(common_part,"bigger_kernel_multiout_NIH",inputs)


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

    return _add_common(common_part,"VGG_multiout_NIH",inputs)


def create_Xception_multioutput(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    base_net = keras.applications.xception.Xception(include_top=False, weights=None,input_shape=[config.img_height,config.img_width,1], pooling="avg")
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"Xception_multiout_NIH",inputs)


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

    return _add_common(common_part,"densenet_multiout_NIH",inputs)


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

    return _add_common(common_part,"resnet_multiout_NIH",inputs)


def create_NASnet_multioutupt(config):
    # load base model
    NASnet_model = keras.applications.NASNetMobile(input_shape=[config.img_height,config.img_width,1], include_top=False,weights=None,)
    
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    common_part = NASnet_model(inputs)
    common_part =  tf.keras.layers.GlobalAveragePooling2D()(common_part)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"NASnet_multiout_NIH",inputs)


# a few models with image net pretrain
# first the ones with 224x224
def create_VGG_multioutput_imagenet(config):
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=[config.img_height,config.img_width,3], pooling="avg")
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"VGG_multiout_NIH_imagenet",inputs)

def create_resnet_multioutput_imagenet(config):
    base_resnet = keras.applications.resnet_v2.ResNet50V2(include_top=False, input_shape=[config.img_height,config.img_width,3],pooling="avg",weights="imagenet",)
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    common_part = base_resnet(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"resnet_multiout_NIH_imagenet",inputs)

def create_densenet_multioutput_imagenet(config):
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg")
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"densenet_multiout_NIH_imagenet",inputs)

def create_NASnet_multioutupt_imagenet(config):
    # load base model
    NASnet_model = keras.applications.NASNetMobile(include_top=False,weights="imagenet",pooling = "avg")
    
    # create new model with common part
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    common_part = NASnet_model(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"NASnet_multiout_NIH_imagenet",inputs)

def create_Xception_multioutput_imagenet(config):

    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,3])
    base_net = keras.applications.xception.Xception(include_top=False, weights="imagenet",input_shape=[config.img_height,config.img_width,3], pooling="avg")
    # create new model with common part
    common_part = base_net(inputs)
    common_part = keras.layers.Dense(512, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)
    common_part = keras.layers.Dense(256, activation='relu')(common_part)
    common_part = keras.layers.BatchNormalization()(common_part)
    common_part = keras.layers.Dropout(0.5)(common_part)

    return _add_common(common_part,"Xception_multiout_NIH_imagenet",inputs)


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

    return _add_common(common_part,"NASnet_large_multiout_NIH",inputs)

def create_relu_joint_res_net(config):
    inputs = keras.layers.Input(shape=[config.img_height,config.img_width,1])
    model = complex_rewritten(inputs, decay = None, use_dense = False)
    
    optimizer = keras.optimizers.Adam(learning_rate = 3e-4)
    
    model = _add_common(model, 'complex_gap_nih', inputs, optimizer = optimizer)
    
    return model

def _add_common(common_part,name,inputs, optimizer = 'adam'):
    disease = keras.layers.Dense(14, activation='sigmoid', name='disease_pred')(common_part)
    sex = keras.layers.Dense(1, activation='sigmoid', name='sex_pred')(common_part)
    age = keras.layers.Dense(1,activation="linear",name="age_pred")(common_part)

    # get final model
    model = keras.models.Model(
        inputs=inputs,
        outputs=[disease, age],
        name=name)

    losses = {
        "disease_pred": "binary_crossentropy",
        # "sex_pred" : "binary_crossentropy",
        "age_pred": "mean_squared_error",
    }
    
    lossWeights = {
        "disease_pred": 2.0, 
        "age_pred": 0.005
    }

    # model.compile(optimizer = optimizer, loss = losses, loss_weights = lossWeights, metrics = {"disease_pred": "binary_accuracy","sex_pred":"binary_accuracy","age_pred":"mae"})
    model.compile(optimizer = optimizer, loss = losses, loss_weights = lossWeights, metrics = {"disease_pred": "binary_accuracy", "age_pred":"mae"})
    
    return model
