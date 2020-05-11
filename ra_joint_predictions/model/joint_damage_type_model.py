import numpy as np
import tensorflow.keras as keras

from model.utils.building_blocks_joints import get_joint_model_input, complex_rewritten
from model.utils.losses import focal_loss
from model.utils.metrics import brier_score

from keras_adamw import AdamW

def load_joint_damage_model(model_file):
    return keras.models.load_model(model_file, compile = False)

def get_joint_damage_type_model(config, optimizer_params, pretrained_model_file = None, model_name = 'joint_damage_type_model', alpha = .9, init_bias = 0):
    base_input, base_ouptut = _get_base_model(config, pretrained_model_file)

    output, metrics_dir, losses = _add_output(base_ouptut, init_bias, alpha)

    joint_damage_type_model = keras.models.Model(
        inputs = base_input,
        outputs = output,
        name = model_name)
    
    optimizer = _get_optimizier(joint_damage_type_model, optimizer_params)
        
    joint_damage_type_model.compile(loss = losses, metrics = metrics_dir, optimizer = optimizer)

    return joint_damage_type_model

def _get_base_model(config, pretrained_model_file):
    if pretrained_model_file is not None:
        pretrained_model = keras.models.load_model(pretrained_model_file)

        return pretrained_model.input, pretrained_model.output
    else:
        input = get_joint_model_input(config)
        base_model = complex_rewritten(input, decay = None, use_dense = False)

        return input, base_model

def _add_output(base_output, init_bias, alpha, gamma = 1.):
    n_bias = init_bias.size

    outputs = []
    losses = []
    metrics_dir = {}

    for n in range(n_bias):
        bias_initializers = keras.initializers.Constant(value = init_bias[n])
        
        outputs.append(keras.layers.Dense(1, activation = 'sigmoid', bias_initializer = bias_initializers, name = f'joint_damage_type_{n}')(base_output))
    
        metrics_dir[f'joint_damage_type_{n}'] = ['binary_accuracy', brier_score]

        losses.append(focal_loss(alpha = alpha[n], gamma = gamma))
    
    return outputs, metrics_dir, losses

def _get_optimizier(model, optimizer_params):
    lr = optimizer_params['lr']
    wd = optimizer_params['wd']
    use_wr = optimizer_params['use_wr']
    total_iterations = optimizer_params['restart_epochs'] * optimizer_params['steps_per_epoch']
    
    weight_decays = {}
    
    # Only layers with "kernel" need wd applied and don't apply WD to the output layer
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel_regularizer = keras.regularizers.l2(0)
            weight_decays.update({layer.kernel.name: wd})
        
    optimizer = AdamW(lr = lr, weight_decays = weight_decays, use_cosine_annealing = use_wr, total_iterations = total_iterations, init_verbose = False, batch_size = 1)
    
    return optimizer
