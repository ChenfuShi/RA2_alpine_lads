import numpy as np

import tensorflow.keras as keras

from model.utils.metrics import argmax_rmse, softmax_rmse_metric, class_softmax_rmse_metric, rmse_metric, class_rmse_metric, mae_metric, class_filter_rmse_metric, softmax_rmse_mae
from model.utils.building_blocks_joints import get_joint_model_input, create_complex_joint_model

def load_joint_damage_model(model_file, no_classes, is_regression = False):
    if not is_regression:
        dependencies = {
            'softmax_rmse': softmax_rmse_metric(np.arange(no_classes)),
            'class_softmax_rmse_0': class_softmax_rmse_metric(np.arange(no_classes), 0),
            'argmax_rmse': argmax_rmse,
            'class_softmax_rsme_0': class_softmax_rmse_metric(np.arange(no_classes), 0), # Added for compatibility with models saved with the previous spelling mistake
            'softmax_rsme': softmax_rmse_metric(np.arange(no_classes)) # Added for compatibility with models saved with the previous spelling mistake
        }
    else:
        dependencies = {
            'rmse': rmse
        }

        for n in range(no_classes):
            dependencies[f'class_{n}_rmse'] = class_rmse_metric(n)

    return keras.models.load_model(model_file, custom_objects = dependencies)

def get_joint_damage_model(config, class_weights, pretrained_model_file = None, model_name = 'joint_damage_model', optimizer = 'adam', is_regression = False):
    base_input, base_ouptut = _get_base_model(config, pretrained_model_file)

    outputs, metrics_dir = _add_outputs(class_weights, base_ouptut, is_regression = is_regression)

    joint_damage_model = keras.models.Model(
        inputs = base_input,
        outputs = outputs,
        name = model_name)

    if not is_regression:
        joint_damage_model.compile(loss = 'categorical_crossentropy', metrics = metrics_dir, optimizer = optimizer)
    else:
        losses = {
            'reg_output_0': 'mean_squared_error',
            'class_output_0': 'categorical_crossentropy'
        }

        lossWeights = {'reg_output_0': 1, 'class_output_0': 1}
        
        joint_damage_model.compile(loss = losses, loss_weights = lossWeights, metrics = metrics_dir, optimizer = optimizer)

    return joint_damage_model

def _get_base_model(config, pretrained_model_file):
    if pretrained_model_file is not None:
        pretrained_model = keras.models.load_model(pretrained_model_file)

        return pretrained_model.input, pretrained_model.output
    else:
        input = get_joint_model_input(config)
        base_model = create_complex_joint_model(input)

        return input, base_model

def _add_outputs(class_weights, base_output, is_regression = False):
    metrics_dir = {}
    outputs = []
    
    for idx, class_weight in enumerate(class_weights):
        no_outcomes = len(class_weight.keys())
        
        reg_metrics = []
        class_metrics = []

        if not is_regression:
            class_metrics.extend([softmax_rmse_mae(np.arange(no_outcomes)), softmax_rmse_metric(np.arange(no_outcomes)), class_softmax_rmse_metric(np.arange(no_outcomes), 0)])
        
            output = keras.layers.Dense(no_outcomes, activation = 'softmax', name = f'class_output_{idx}')(base_output)
            outputs.append(output)
        else:
            max_outcome = max(class_weight.keys())
            
            req_output = keras.layers.Dense(1, activation = 'linear', name = f'reg_output_{idx}')(base_output)
            class_output = keras.layers.Dense(no_outcomes, activation = 'softmax', name = f'class_output_{idx}')(base_output)
            
            reg_metrics.append(mae_metric(max_outcome))
            reg_metrics.append(rmse_metric(max_outcome))
            reg_metrics.append(class_filter_rmse_metric(max_outcome, 0))
            
            class_metrics.extend([softmax_rmse_mae(np.arange(no_outcomes)), softmax_rmse_metric(np.arange(no_outcomes)), class_softmax_rmse_metric(np.arange(no_outcomes), 0)])
                
            outputs.append(req_output)
            outputs.append(class_output)
        
        metrics_dir[f'reg_output_{idx}'] = reg_metrics
        metrics_dir[f'class_output_{idx}'] = class_metrics

    return outputs, metrics_dir
