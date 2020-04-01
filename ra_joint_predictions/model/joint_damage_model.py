import numpy as np

import tensorflow.keras as keras

from model.utils.metrics import mae_metric, rmse_metric, class_filter_rmse_metric, softmax_mae_metric, softmax_rmse_metric, class_filter_softmax_rmse_metric

from model.utils.building_blocks_joints import get_joint_model_input, create_complex_joint_model

from keras_adamw import AdamW

MODEL_TYPE_CLASSIFICATION = "C"
MODEL_TYPE_REGRESSION = "R"
MODEL_TYPE_COMBINED = "RC"

def load_joint_damage_model(model_file):
    return keras.models.load_model(model_file, compile = False)

def get_joint_damage_model(config, class_weights, pretrained_model_file = None, model_name = 'joint_damage_model', optimizer = 'adam', model_type = 'R'):
    base_input, base_ouptut = _get_base_model(config, pretrained_model_file)

    outputs, metrics_dir = _add_outputs(class_weights, base_ouptut, model_type = model_type)

    joint_damage_model = keras.models.Model(
        inputs = base_input,
        outputs = outputs,
        name = model_name)
    
    optimizer = _get_optimizier(joint_damage_model)

    if model_type == MODEL_TYPE_CLASSIFICATION:
        joint_damage_model.compile(loss = 'categorical_crossentropy', metrics = metrics_dir, optimizer = optimizer)
    elif model_type == MODEL_TYPE_REGRESSION:
        joint_damage_model.compile(loss = 'mean_squared_error', metrics = metrics_dir, optimizer = optimizer)
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

def _get_base_model(config, pretrained_model_file):
    if pretrained_model_file is not None:
        pretrained_model = keras.models.load_model(pretrained_model_file)

        #for layer in pretrained_model.layers[:-10]:
            #layer.trainable = False
        
        return pretrained_model.input, pretrained_model.output
    else:
        input = get_joint_model_input(config)
        base_model = create_complex_joint_model(input)

        return input, base_model

def _add_outputs(class_weights, base_output, model_type):
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

def _get_optimizier(model):
    weight_decays = {}

    for layer in model.layers:
        layer.kernel_regularizer = keras.regularizers.l2(0)
        weight_decays.update({layer.name: 1e-4})

    optimizer = AdamW(lr = 3e-4, weight_decays = weight_decays, use_cosine_annealing = True, total_iterations = 10 * 60, init_verbose = False)
    
    return optimizer