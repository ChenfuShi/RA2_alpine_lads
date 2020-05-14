import logging

import numpy as np
import tensorflow.keras as keras
import tensorflow_addons as tfa

from keras_adamw import AdamW, get_weight_decays, fill_dict_in_order

from model.utils.building_blocks_joints import get_joint_model_input, create_complex_joint_model, complex_rewritten, _vvg_fc_block
from model.utils.metrics import mae_metric, rmse_metric, class_filter_rmse_metric, softmax_mae_metric, softmax_rmse_metric, class_filter_softmax_rmse_metric
from model.utils.layers import ReLUOutput
from model.utils.losses import softmax_focal_loss, pseudo_huber_loss

MODEL_TYPE_CLASSIFICATION = "C"
MODEL_TYPE_REGRESSION = "R"
MODEL_TYPE_COMBINED = "RC"

def load_joint_damage_model(model_file):
    return keras.models.load_model(model_file, compile = False)

def get_joint_damage_model(config, class_weights, params, pretrained_model_file = None, model_name = 'joint_damage_model', model_type = 'R', has_outputs = False):
    base_input, base_ouptut = _get_base_model(config, pretrained_model_file)

    outputs, metrics_dir = _add_outputs(class_weights, base_ouptut, model_type, params.get('is_wrist', False))

    if has_outputs:
        outputs = [base_ouptut]
    
    joint_damage_model = keras.models.Model(
        inputs = base_input,
        outputs = outputs,
        name = model_name)
    
    optimizer = _get_optimizier(joint_damage_model, params)

    if model_type == MODEL_TYPE_CLASSIFICATION:
        joint_damage_model.compile(loss = 'categorical_crossentropy', metrics = metrics_dir, optimizer = optimizer)
    elif model_type == MODEL_TYPE_REGRESSION:
        group_flag = params.get('group_flag', None)
        
        loss_weights = None
    
        if group_flag is not None:
            if group_flag == 'R':
                loss_weights = [60 / 3, 60 / 3, 40 / 3, 60 / 3, 40 / 3, 40 / 3]
            elif group_flag == 'L':
                loss_weights = [40 / 3, 40 / 3, 60 / 3, 40 / 3, 60 / 3, 60 / 3]
                
            logging.info('Loss Weights:', loss_weights)
        
        joint_damage_model.compile(loss = 'mean_squared_error', metrics = metrics_dir, optimizer = optimizer, loss_weights = loss_weights)
    elif model_type == MODEL_TYPE_COMBINED:
        losses = {}
        lossWeights = {}

        for n in range(0, len(outputs), 2):
            losses[f'reg_output_{n}'] = 'mean_squared_error'
            losses[f'reg_output_{n + 1}'] = 'categorical_crossentropy'

            lossWeights[f'reg_output_{n}'] = 1
            lossWeights[f'class_output_{n + 1}'] = 1
        
        joint_damage_model.compile(loss = losses, loss_weights = lossWeights, metrics = metrics_dir, optimizer = optimizer)

    return joint_damage_model

def load_minority_model(model_file, class_weights, epochs, steps, model_name = 'finetuned_joint_damage_model'):
    joint_damage_model = keras.models.load_model(model_file, compile = False)
    
    idx = 0
    max_outcome = max(class_weights[0].keys())
    metrics_dir = {}
    metrics_dir[f'reg_output_{idx}'] = [mae_metric(max_outcome), rmse_metric(max_outcome), class_filter_rmse_metric(max_outcome, 0)]
    
    lr_decay = keras.experimental.CosineDecay(3e-4, epochs * steps, alpha = 1/3)
    
    # lr_decay = keras.experimental.CosineDecayRestarts(5e-4, 25 * 120, m_mul = 0.9, alpha = 0.1)
    optimizer = keras.optimizers.SGD(learning_rate = 1e-2, momentum = 0.9)
    
    joint_damage_model.compile(loss = 'mean_squared_error', metrics = metrics_dir, optimizer = optimizer)
    
    return joint_damage_model

def _get_base_model(config, pretrained_model_file):
    if pretrained_model_file is not None:
        pretrained_model = keras.models.load_model(pretrained_model_file, compile = False)
        
        return pretrained_model.input, pretrained_model.output
    else:
        input = get_joint_model_input(config)
        base_model = complex_rewritten(input, decay = None, use_dense = False)

        return input, base_model

def _add_outputs(class_weights, base_output, model_type, is_wrist):
    metrics_dir = {}
    outputs = []
    
    for idx, class_weight in enumerate(class_weights):
        no_outcomes = len(class_weight.keys())

        if 'R' in model_type:
            req_output = keras.layers.Dense(1, activation = 'linear', name = f'reg_output_{idx}')(base_output)
            outputs.append(req_output)
            
            max_outcome = max(class_weight.keys())

            metrics_dir[f'reg_output_{idx}'] = [mae_metric(max_outcome), rmse_metric(max_outcome), class_filter_rmse_metric(max_outcome, 0)]
        
        if 'C' in model_type:
            [softmax_mae_metric(np.arange(no_outcomes)), softmax_rmse_metric(np.arange(no_outcomes)), class_filter_softmax_rmse_metric(np.arange(no_outcomes), 0)]

            output = keras.layers.Dense(no_outcomes, activation = 'softmax', name = f'class_output_{idx}')(base_output)
            outputs.append(output)

            metrics_dir[f'class_output_{idx}'] = [softmax_mae_metric(np.arange(no_outcomes)), softmax_rmse_metric(np.arange(no_outcomes)), class_filter_softmax_rmse_metric(np.arange(no_outcomes), 0)]

    return outputs, metrics_dir

def _get_optimizier(model, params):
    epochs = params['epochs']
    steps_per_epoch = params['steps_per_epoch']
    lr = params['lr']
    wd = params['wd']
    
    weight_decays = {}
    
    # Only layers with "kernel" need wd applied
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel_regularizer = keras.regularizers.l2(0)
            weight_decays.update({layer.kernel.name: wd})
            
    optimizer = AdamW(lr = lr, use_cosine_annealing = True, weight_decays = weight_decays, total_iterations = epochs * steps_per_epoch, init_verbose = False, batch_size = 1)
    
    return optimizer

