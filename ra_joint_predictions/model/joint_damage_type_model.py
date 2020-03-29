import numpy as np
import tensorflow.keras as keras

from model.utils.building_blocks_joints import get_joint_model_input, create_complex_joint_model
from model.utils.losses import focal_loss
from model.utils.metrics import brier_score

def load_joint_damage_model(model_file):
    return keras.models.load_model(model_file, compile = False)

def get_joint_damage_type_model(config, pretrained_model_file = None, model_name = 'joint_damage_type_model', optimizer = 'adam', alpha = .9):
    base_input, base_ouptut = _get_base_model(config, pretrained_model_file)

    output, metrics_dir = _add_output(base_ouptut)

    joint_damage_type_model = keras.models.Model(
        inputs = base_input,
        outputs = output,
        name = model_name)
        
    joint_damage_type_model.compile(loss = focal_loss(alpha = alpha), metrics = metrics_dir, optimizer = optimizer)

    return joint_damage_type_model

def _get_base_model(config, pretrained_model_file):
    if pretrained_model_file is not None:
        pretrained_model = keras.models.load_model(pretrained_model_file)

        return pretrained_model.input, pretrained_model.output
    else:
        input = get_joint_model_input(config)
        base_model = create_complex_joint_model(input)

        return input, base_model

def _add_output(base_output):
    output = keras.layers.Dense(1, activation = 'sigmoid', name = 'joint_damage_type')(base_output)
    
    metrics = ['binary_accuracy', brier_score]
    
    return output, metrics