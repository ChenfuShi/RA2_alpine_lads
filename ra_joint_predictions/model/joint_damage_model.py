import numpy as np

import tensorflow.keras as keras

from model.utils.metrics import argmax_rsme, softmax_rsme_metric, class_softmax_rsme_metric
from model.utils.building_blocks_joints import get_joint_model_input, create_complex_joint_model

def load_joint_damage_model(model_file, no_classes):
    dependencies = {
        'softmax_rsme': softmax_rsme_metric(np.arange(no_classes)),
        'class_softmax_rmse_0': class_softmax_rsme_metric(np.arange(no_classes), 0),
        'argmax_rsme': argmax_rsme    
    }

    return keras.models.load_model(model_file, custom_objects = dependencies)

def get_joint_damage_model(config, class_weights, pretrained_model_file = None, model_name = 'joint_damage_model', optimizer = 'adam'):
    base_input, base_ouptut = _get_base_model(config, pretrained_model_file)

    outputs, metrics_dir = _add_outputs(class_weights, base_ouptut)

    joint_damage_model = keras.models.Model(
        inputs = base_input,
        outputs = outputs,
        name = model_name)

    joint_damage_model.compile(loss = 'categorical_crossentropy', metrics = metrics_dir, optimizer = optimizer)

    return joint_damage_model

def _get_base_model(config, pretrained_model_file):
    if pretrained_model_file is not None:
        pretrained_model = keras.models.load_model(pretrained_model_file)

        return pretrained_model.input, pretrained_model.output
    else:
        input = get_joint_model_input(config)
        base_model = create_complex_joint_model(input)

        return input, base_model

def _add_outputs(class_weights, base_output):
    metrics_dir = {}
    outputs = []
    
    for idx, class_weight in enumerate(class_weights):
        no_outcomes = len(class_weight.keys())
        
        metrics = [softmax_rsme_metric(np.arange(no_outcomes)), class_softmax_rsme_metric(np.arange(no_outcomes), 0)]
        
        output = keras.layers.Dense(no_outcomes, activation = 'softmax', name = f'output_{idx}')(base_output)
        outputs.append(output)
        
        metrics_dir[f'output_{idx}'] = metrics

    return outputs, metrics_dir
