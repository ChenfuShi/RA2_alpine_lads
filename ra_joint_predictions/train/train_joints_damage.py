import datetime
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.backend as K

import dataset.joint_dataset as joint_dataset

from dataset.joint_val_dataset import hands_joints_val_dataset, hands_wrists_val_dataset, feet_joint_val_dataset, combined_joint_val_dataset
from dataset.test_dataset import joint_test_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor
from model.joint_damage_model import get_joint_damage_model, load_minority_model
from utils.saver import CustomSaver, _get_tensorboard_callback
from model.utils.metrics import mae_metric, rmse_metric, class_filter_rmse_metric
from train.utils.callbacks import AdamWWarmRestartCallback

train_params = {
    'epochs': 300,
    'batch_size': 64,
    'steps_per_epoch': 130
}

finetune_params = {
    'epochs': 50,
    'batch_size': 64,
    'steps_per_epoch': 160
}

def train_joints_damage_model(config, model_name, pretrained_model, joint_type, dmg_type, do_validation = False, model_type = 'R'):
    logging.info(f'Training model with joint_type: {joint_type} - dmg_type: {dmg_type}')
    
    joint_dataset, non0_tf_dataset, tf_joint_val_dataset, no_val_samples = _get_dataset(config, joint_type, dmg_type, model_type, do_validation = do_validation, split_type = 'balanced')
    
    logging.info('Class Weights: %s', joint_dataset.class_weights)
    
    params = train_params.copy()
    
    if joint_type == 'H' and dmg_type == 'J':
        params['steps_per_epoch'] = 115
    elif joint_type == 'F' and dmg_type == 'E':
        params['steps_per_epoch'] = 80
    elif joint_type == 'F' and dmg_type == 'J':
        params['steps_per_epoch'] = 80
    elif joint_type == 'HF' and dmg_type == 'J':
        params['steps_per_epoch'] = 200
    
    model = get_joint_damage_model(config, joint_dataset.class_weights, params['epochs'], params['steps_per_epoch'], pretrained_model_file = pretrained_model, model_name = model_name, model_type = model_type)

    return _fit_joint_damage_model(model, model.name, non0_tf_dataset, joint_dataset.class_weights, params, tf_joint_val_dataset, no_val_samples)

def finetune_minority_model(config, model_name, minority_model, joint_type, dmg_type, do_validation = False, model_type = 'R'):
    joint_dataset, tf_joint_dataset, tf_joint_val_dataset, no_val_samples = _get_dataset(config, joint_type, dmg_type, model_type, do_validation = do_validation)
    
    params = finetune_params.copy()
    
    if joint_type == 'H' and dmg_type == 'J':
        params['steps_per_epoch'] = 135
    elif joint_type == 'F' and dmg_type == 'E':
        params['steps_per_epoch'] = 95
    elif joint_type == 'F' and dmg_type == 'J':
        params['steps_per_epoch'] = 95
    
    model = load_minority_model(minority_model, joint_dataset.class_weights, params['epochs'], params['steps_per_epoch'], model_name = model_name)
    
    def scheduler(epoch):
        if epoch < 60:
            return 15e-5
        if epoch < 120:
            return 1e-4
        else:
            return 1e-4
        
    params['scheduler'] = scheduler
    
    return _fit_joint_damage_model(model, model.name + '_finetune', tf_joint_dataset, joint_dataset.class_weights, params, tf_joint_val_dataset, no_val_samples)

def _get_dataset(config, joint_type, dmg_type, model_type, do_validation = False, split_type = None):
    outcomes_source = os.path.join(config.train_location, 'training.csv')
    
    apply_clahe = False
    
    if not do_validation:
        tf_val_dataset = None
        no_val_samples = 0

    erosion_flag = dmg_type == 'E'
    
    df_joint_extractor = get_joint_extractor(joint_type, erosion_flag)

    if joint_type == 'F':
        joint_dataset = feet_joint_val_dataset(config, model_type = model_type, pad_resize = False, joint_extractor = df_joint_extractor, split_type = split_type)

        if do_validation:
            tf_dataset, tf_val_dataset, no_val_samples = joint_dataset.create_feet_joints_dataset_with_validation(outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        else:
            tf_dataset = joint_val_dataset.create_feet_joints_dataset(outcomes_source = outcomes_source, erosion_flag = erosion_flag)
    elif joint_type == 'H':
        joint_dataset = hands_joints_val_dataset(config, model_type = model_type, pad_resize = False, joint_extractor = df_joint_extractor, imagenet = False, split_type = split_type)
        
        if do_validation:
            tf_dataset, tf_val_dataset, no_val_samples = joint_dataset.create_hands_joints_dataset_with_validation(outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        else:
            tf_dataset = joint_dataset.create_hands_joints_dataset(outcomes_source = outcomes_source, erosion_flag = erosion_flag)
    elif joint_type == 'W':
        joint_dataset = hands_wrists_val_dataset(config, model_type = model_type, pad_resize = False, imagenet = False)

        if do_validation:
            tf_dataset, tf_val_dataset, no_val_samples = joint_dataset.create_wrists_joints_dataset_with_validation(outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        else:
            tf_dataset = joint_dataset.create_wrists_joints_dataset(outcomes_source = outcomes_source, erosion_flag = erosion_flag)

    elif joint_type == 'HF' and not erosion_flag:
        joint_dataset = combined_joint_val_dataset(config, model_type = model_type, pad_resize = False, joint_extractor = df_joint_extractor)

        if do_validation:
            tf_dataset, tf_val_dataset, no_val_samples = joint_dataset.create_combined_joint_dataset_with_validation(outcomes_source)
        else:
            tf_dataset = joint_dataset.create_combined_joint_dataset(outcomes_source = outcomes_source)
            
    return joint_dataset, tf_dataset, tf_val_dataset, no_val_samples

def _fit_joint_damage_model(model, model_name, tf_joint_dataset, class_weights, params, tf_joint_val_dataset = None, no_val_samples = 0):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)
    
    logging.info('_fit_joint_damage_model params %s', params)
    
    epochs = params['epochs']
    steps_per_epoch = params['steps_per_epoch']
    batch_size = params['batch_size']
    
    if tf_joint_val_dataset is None:
        history = model.fit(
            tf_joint_dataset, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2, callbacks = [saver, tensorboard_callback])
    else:
        val_steps = np.ceil(no_val_samples / batch_size)
        
        history = model.fit(tf_joint_dataset, 
            epochs = epochs, steps_per_epoch = steps_per_epoch, validation_data = tf_joint_val_dataset, validation_steps = val_steps, verbose = 2, callbacks = [saver, tensorboard_callback])

    hist_df = pd.DataFrame(history.history)

    return model, hist_df
