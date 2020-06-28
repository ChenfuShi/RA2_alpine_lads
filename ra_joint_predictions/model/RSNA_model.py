import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import model.NIH_model as NIH
from model.utils.building_blocks_joints import _fc_block
from model.utils.building_blocks_joints import create_complex_joint_model, relu_joint_res_net, extended_complex
from model.utils.building_blocks_joints import get_joint_model_input, vvg_joint_model, complex_rewritten, avg_joint_vgg, vgg19_with_sp_dropout, small_resnet_with_bottleneck, small_densenet, bottlenecked_small_dense, small_vgg_with_bottleneck, small_vgg_with_bottleneck

from model import NIH_model

from keras_adamw import AdamW

elu_activation = lambda x: keras.activations.elu(x, alpha = 0.3)

def complex_joint_finetune_model(*args, **kwargs):
    return model_finetune_RSNA(*args, **kwargs)

def model_finetune_RSNA(config, no_joint_types = 10, weights = "weights/NIH_new_pretrain_model_100.h5", name = "rsna_complex_multiout",act="relu"):
    
    if act != "relu":
        NIH_model = keras.models.load_model(weights,custom_objects = {"<lambda>" : (lambda x: keras.activations.elu(x, alpha = 0.1))})
    else:
        NIH_model = keras.models.load_model(weights)

    if "rewritten" in weights:
        NEW_model = keras.Model(NIH_model.layers[1].input, NIH_model.layers[-4].output)
    else:
        NEW_model = keras.Model(NIH_model.input, NIH_model.layers[-4].output)

    # this is blocking the fully connected as well...
    for layer in NEW_model.layers:
        layer.trainable=False

    # split into three parts
    
    boneage_fc = keras.layers.Dense(32, activation = act, name = 'boneage_1' + '_fc')(NEW_model.output)
    boneage_fc = keras.layers.BatchNormalization(name = 'boneage_1' + '_batch')(boneage_fc)
    boneage_fc = keras.layers.Dropout(0.5, name = 'boneage_1' + '_dropout')(boneage_fc)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(boneage_fc)

    sex_fc = keras.layers.Dense(32, activation = act, name = 'sex_1' + '_fc')(NEW_model.output)
    sex_fc = keras.layers.BatchNormalization(name = 'sex_1' + '_batch')(sex_fc)
    sex_fc = keras.layers.Dropout(0.5, name = 'sex_1' + '_dropout')(sex_fc)
    
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(sex_fc)

    joint_type_fc = keras.layers.Dense(64, activation = act, name = 'joint_type_1' + '_fc')(NEW_model.output)
    joint_type_fc = keras.layers.BatchNormalization(name = 'joint_type_1' + '_batch')(joint_type_fc)
    joint_type_fc = keras.layers.Dropout(0.5, name = 'joint_type_1' + '_dropout')(joint_type_fc)
    
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred')(joint_type_fc)

    # get final model
    model = keras.models.Model(
        inputs=NEW_model.input,
        outputs=[boneage, sex, joint_type],
        name=name)

    losses = {
        'boneage_pred': 'mean_squared_error',
        'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': 'categorical_crossentropy',
    }

    lossWeights = {'boneage_pred': 0.005, 'sex_pred': 0, 'joint_type_pred': 1}

    model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'categorical_accuracy'})

    return model

def create_vgg_rsna_model(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = complex_rewritten(input_layer)

    boneage_fc = keras.layers.Dense(32, name = 'boneage_1' + '_fc')(model)
    boneage_fc = keras.layers.ELU()(boneage_fc)
    boneage_fc = keras.layers.BatchNormalization(name = 'boneage_1' + '_batch')(boneage_fc)
    boneage_fc = keras.layers.Dropout(0.5, name = 'boneage_1' + '_dropout')(boneage_fc)
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred')(boneage_fc)

    sex_fc = keras.layers.Dense(32, name = 'sex_1' + '_fc')(model)
    sex_fc = keras.layers.ELU()(sex_fc)
    sex_fc = keras.layers.BatchNormalization(name = 'sex_1' + '_batch')(sex_fc)
    sex_fc = keras.layers.Dropout(0.5, name = 'sex_1' + '_dropout')(sex_fc)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred')(sex_fc)

    joint_type_fc = keras.layers.Dense(64, name = 'joint_type_1' + '_fc')(model)
    joint_type_fc = keras.layers.ELU()(joint_type_fc)
    joint_type_fc = keras.layers.BatchNormalization(name = 'joint_type_1' + '_batch')(joint_type_fc)
    joint_type_fc = keras.layers.Dropout(0.5, name = 'joint_type_1' + '_dropout')(joint_type_fc)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred')(joint_type_fc)

    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)

def create_complex(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = complex_rewritten(input_layer, decay = None, use_dense = False)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)

def create_res_rsna_model(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = relu_joint_res_net(input_layer, decay = None)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)

def create_rsna_extended_complex(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = extended_complex(input_layer)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)
    
def create_vgg_avg_rsna_model(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = avg_joint_vgg(input_layer, decay = None)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name) 
    
def create_vgg19_with_sp_dropout(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = vgg19_with_sp_dropout(input_layer)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)

def create_small_bottlenecked_vgg(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    
    model = small_vgg_with_bottleneck(input_layer)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)

def create_small_resnet_withbottleneck(config, name, no_joint_types = 13):
    input_layer = get_joint_model_input(config)
    model = small_resnet_with_bottleneck(input_layer)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)

def create_small_densenet(config, name, no_joint_types = 13, input_layer = None):
    if input_layer is None:
        input_layer = get_joint_model_input(config)
        
    model = bottlenecked_small_dense(input_layer)
    
    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name)
    
def finetune_rsna_model(config, weights, name, no_joint_types = 13):
    pretrained_model = keras.models.load_model(weights, compile = False)
    
    input_layer = keras.layers.Input(shape = [192, 256, 1])
    
    new_model = NIH_model.create_complex_joint_multioutput(config, name, inputs = input_layer)
    new_model.set_weights(pretrained_model.get_weights())
    
    model_output = new_model.layers[-3].output
    
    #if "rewritten" in weights:
    # input_layer = pretrained_model.layers[1].input
    #else:
        #input_layer = pretrained_model.input


    boneage = keras.layers.Dense(1, activation = 'linear', name = 'boneage_pred', kernel_initializer = 'he_uniform')(model_output)
    sex = keras.layers.Dense(1, activation = 'sigmoid', name = 'sex_pred', kernel_initializer = 'he_uniform')(model_output)
    joint_type = keras.layers.Dense(no_joint_types, activation = 'softmax', name = 'joint_type_pred', kernel_initializer = 'he_uniform')(model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate = 3e-4)
    
    return _create_compile_rsna_multioutput(input_layer, boneage, sex, joint_type, name, optimizier = optimizer) 

def _create_compile_rsna_multioutput(input, boneage, sex, joint_type, name, optimizier = 'adam'):
    # get final model
    model = keras.models.Model(
        inputs=input,
        outputs=[boneage, joint_type],
        name=name)
    
    for layer in model.layers[:-2]:
        if hasattr(layer, 'kernel'):
                layer.trainable = False

    losses = {
        'boneage_pred': 'mean_squared_error',
        # 'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': 'categorical_crossentropy',
    }

    lossWeights = {'boneage_pred': 0.005, 'joint_type_pred': 1}
    
    model.compile(optimizer = optimizier, loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'joint_type_pred': 'categorical_accuracy'})

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



