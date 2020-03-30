import os
import logging

import numpy as np
import pandas as pd
import tensorflow.keras as keras

from dataset.joint_damage_type_dataset import joint_damage_type_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor
from dataset.test_dataset import joint_test_dataset
from model.joint_damage_type_model import get_joint_damage_type_model
from utils.saver import CustomSaver, _get_tensorboard_callback


train_params = {
    'epochs': 50,
    'batch_size': 64,
    'steps_per_epoch': 125
}

def train_joints_damage_type_model(config, model_name, pretrained_model, joint_type, dmg_type, do_validation = False):
    tf_dataset, alpha, tf_val_dataset, val_no_samples = _get_dataset(config, joint_type, dmg_type, do_validation)

    optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
    model = get_joint_damage_type_model(config, pretrained_model, model_name = model_name, optimizer = optimizer, alpha = alpha)

    return _fit_joints_damage_type_model(model, tf_dataset, train_params, val_dataset = tf_val_dataset, no_val_samples = val_no_samples)

def _get_dataset(config, joint_type, dmg_type, do_validation):
    outcomes_source = os.path.join(config.train_location, 'training.csv')
    df_joint_extractor = get_joint_extractor(joint_type, dmg_type)

    dataset = joint_damage_type_dataset(config, pad_resize = False, joint_extractor = joint_extractor)

    erosion_flag = dmg_type == 'E'

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

    alpha = dataset.alpha
            
    return tf_dataset, alpha, tf_val_dataset, val_no_samples
    

def _fit_joints_damage_type_model(model, dataset, train_params, val_dataset = None, no_val_samples = 0):
    saver = CustomSaver(model.name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model.name)
    
    epochs = train_params['epochs']
    steps_per_epoch = train_params['steps_per_epoch']
    batch_size = train_params['batch_size']

    if val_dataset is None:
        history = model.fit(
            dataset, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2, callbacks = [saver, tensorboard_callback])
    else:
        val_steps = np.ceil(no_val_samples / batch_size)
        
        history = model.fit(
            dataset, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2, callbacks = [saver, tensorboard_callback],
                validation_data = val_dataset, validation_steps = val_steps)

    hist_df = pd.DataFrame(history.history)

    return model, hist_df