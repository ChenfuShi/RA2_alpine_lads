import os
import logging

import numpy as np
import pandas as pd
import tensorflow.keras as keras

from dataset.joint_damage_type_dataset import joint_damage_type_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor
from dataset.test_dataset import joint_test_dataset
from model.joint_damage_type_model import get_joint_damage_type_model
from train.utils.callbacks import AdamWWarmRestartCallback
from utils.saver import CustomSaver, _get_tensorboard_callback

train_params = {
    'epochs': 75,
    'batch_size': 64,
    'restart_epochs': 0,
    'lr': 3e-4,
    'wd': 1e-6
}

def train_joints_damage_type_model(config, model_name, pretrained_model, joint_type, dmg_type, do_validation = False):
    tf_dataset, N, alpha, init_bias, tf_val_dataset, val_no_samples = _get_dataset(config, joint_type, dmg_type, do_validation)
    
    params = train_params.copy()
    
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    restart_epochs = params['restart_epochs']
    # If warm restart
    if restart_epochs > 0:
        adamW_warm_restart_callback = AdamWWarmRestartCallback(restart_epochs = restart_epochs)
        params['use_wr'] = True
    else:
        adamW_warm_restart_callback = None
        params['use_wr'] = True
        params['restart_epochs'] = epochs
    
    # Normalize steps to always pass through the dataset exactly once per epoch
    steps_per_epoch = np.ceil(N / batch_size)
    params['steps_per_epoch'] = steps_per_epoch
    
    model = get_joint_damage_type_model(config, params, pretrained_model, model_name = model_name, alpha = alpha, init_bias = init_bias)

    return _fit_joints_damage_type_model(model, tf_dataset, params, val_dataset = tf_val_dataset, no_val_samples = val_no_samples, wr_callback = adamW_warm_restart_callback)

def _get_dataset(config, joint_type, dmg_type, do_validation):
    apply_clahe = False
    
    outcomes_source = os.path.join(config.train_location, 'training.csv')

    erosion_flag = dmg_type == 'E'
    joint_extractor = get_joint_extractor(joint_type, erosion_flag)
    
    dataset = joint_damage_type_dataset(config, pad_resize = False, joint_extractor = joint_extractor, apply_clahe = apply_clahe)

    tf_val_dataset = None
    val_no_samples = 0

    if joint_type == 'H':
        if do_validation:
            tf_dataset, tf_val_dataset, val_no_samples = dataset.get_hands_joint_damage_type_dataset_with_validation(outcomes_source, erosion_flag = erosion_flag)
        else:
            tf_dataset = dataset.get_hands_joint_damage_type_dataset(outcomes_source, erosion_flag = erosion_flag)
    elif joint_type == 'F':
        if do_validation:
            tf_dataset, tf_val_dataset, val_no_samples = dataset.get_feet_joint_damage_type_dataset_with_validation(outcomes_source, erosion_flag = erosion_flag)
        else:
            tf_dataset = dataset.get_feet_joint_damage_type_dataset(outcomes_source, erosion_flag = erosion_flag)
    elif joint_type == 'HF':
        if do_validation:
            tf_dataset, tf_val_dataset, val_no_samples = dataset.get_combined_joint_damage_type_dataset_with_validation(outcomes_source, erosion_flag = erosion_flag)
        else:
            tf_dataset = dataset.get_combined_joint_damage_type_dataset(outcomes_source, erosion_flag = erosion_flag)

    N = dataset.outcomes.shape[0]
    alpha = dataset.alpha
    init_bias = np.log(dataset.n_positives/dataset.n_negatives)
            
    return tf_dataset, N, alpha, init_bias, tf_val_dataset, val_no_samples

def _fit_joints_damage_type_model(model, dataset, train_params, val_dataset = None, no_val_samples = 0, wr_callback = None):
    epochs = train_params['epochs']
    batch_size = train_params['batch_size']
    steps_per_epoch = train_params['steps_per_epoch']
    
    callbacks = [CustomSaver(model.name, n = 5), _get_tensorboard_callback(model.name, log_dir = '../logs/tensorboard/joint_damage_type/')]
    if wr_callback is not None:
        callbacks.append(wr_callback)

    if val_dataset is None:
        history = model.fit(
            dataset, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2, callbacks = callbacks)
    else:
        val_steps = np.ceil(no_val_samples / batch_size)
        
        history = model.fit(
            dataset, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2, callbacks = callbacks,
                validation_data = val_dataset, validation_steps = val_steps)

    hist_df = pd.DataFrame(history.history)

    return model, hist_df