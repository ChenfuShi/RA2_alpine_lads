import datetime
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras

from dataset.joint_dataset import feet_joint_dataset, hands_joints_dataset, hands_wrists_dataset, joint_narrowing_dataset
from dataset.test_dataset import joint_test_dataset, narrowing_test_dataset
from model.joint_damage_model import get_joint_damage_model
from utils.saver import CustomSaver, _get_tensorboard_callback

train_params = {
    'epochs': 300,
    'batch_size': 64,
    'steps_per_epoch': 125
}

def train_joints_damage_model(config, model_name, pretrained_model, joint_type, dmg_type, do_validation = False, model_type = 'R'):
    joint_dataset, tf_joint_dataset, tf_joint_val_dataset, no_val_samples = _get_dataset(config, joint_type, dmg_type, model_type, do_validation = do_validation)
    logging.info('Class Weights: %s', joint_dataset.class_weights)

    # optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.75, nesterov = True)
    optimizer = 'adam'
    model = get_joint_damage_model(config, joint_dataset.class_weights, pretrained_model, model_name = model_name, optimizer = optimizer, model_type = model_type)

    params = train_params.copy()
    if joint_type == 'W':
        params['steps_per_epoch'] = 75
    elif joint_type == 'HF':
        params['steps_per_epoch'] = 175

    return _fit_joint_damage_model(model, tf_joint_dataset, joint_dataset.class_weights, params, tf_joint_val_dataset, no_val_samples)

def _get_dataset(config, joint_type, dmg_type, model_type, do_validation = False):
    outcomes_source = os.path.join(config.train_location, 'training.csv')

    if do_validation:
        hand_joints_source = './data/predictions/hand_joint_data_train_v2.csv'
        feet_joints_source = './data/predictions/feet_joint_data_train_v2.csv'
        hand_joints_val_source = './data/predictions/hand_joint_data_test_v2.csv'
        feet_joints_val_source = './data/predictions/feet_joint_data_test_v2.csv'

        val_dataset = joint_test_dataset(config, config.train_fixed_location, model_type = model_type, pad_resize = True, joint_scale = 5)
    else:
        hand_joints_source = './data/predictions/hand_joint_data_v2.csv'
        feet_joints_source = './data/predictions/feet_joint_data_v2.csv'
        hand_joints_val_source = None
        feet_joints_val_source = None

        tf_val_dataset = None
        no_samples = 0

    erosion_flag = dmg_type == 'E'
    
    if joint_type == 'F':
        joint_dataset = feet_joint_dataset(config, model_type = model_type, pad_resize = True, joint_scale = 5)
        tf_dataset = joint_dataset.create_feet_joints_dataset(outcomes_source, joints_source = feet_joints_source, erosion_flag = erosion_flag)

        if do_validation:
            tf_val_dataset, no_samples = val_dataset.get_feet_joint_test_dataset(feet_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

    elif joint_type == 'H':
        joint_dataset = hands_joints_dataset(config, model_type = model_type, pad_resize = True, joint_scale = 5)
        tf_dataset = joint_dataset.create_hands_joints_dataset(outcomes_source, joints_source = hand_joints_source, erosion_flag = erosion_flag)

        if do_validation:
            tf_val_dataset, no_samples = val_dataset.get_hands_joint_test_dataset(hand_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

    elif joint_type == 'W':
        joint_dataset = hands_wrists_dataset(config, model_type = model_type)
        tf_dataset = joint_dataset.create_wrists_joints_dataset(outcomes_source, joints_source = hand_joints_source, erosion_flag = erosion_flag)

        if do_validation:
            tf_val_dataset, no_samples = val_dataset.get_wrists_joint_test_dataset(hand_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

    elif joint_type == 'HF' and not erosion_flag:
        joint_dataset = joint_narrowing_dataset(config, model_type = model_type, pad_resize = True, joint_scale = 5)
        tf_dataset = joint_dataset.create_combined_narrowing_joint_dataset(outcomes_source, hand_joints_source = hand_joints_source, feet_joints_source = feet_joints_source)

        if do_validation:
            val_dataset = narrowing_test_dataset(config, config.train_fixed_location, model_type = model_type)

            tf_val_dataset, no_samples = val_dataset.get_joint_narrowing_test_dataset(hand_joints_source = hand_joints_val_source, feet_joints_source = feet_joints_val_source, outcomes_source = outcomes_source)

    return joint_dataset, tf_dataset, tf_val_dataset, no_samples

def _fit_joint_damage_model(model, tf_joint_dataset, class_weights, train_params, tf_joint_val_dataset = None, no_val_samples = 0):
    saver = CustomSaver(model.name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model.name)

    epochs = train_params['epochs']
    steps_per_epoch = train_params['steps_per_epoch']
    batch_size = train_params['batch_size']

    if tf_joint_val_dataset is None:
        history = model.fit(
            tf_joint_dataset, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2, callbacks = [saver, tensorboard_callback])
    else:
        val_steps = np.ceil(no_val_samples / batch_size)

        history = model.fit(tf_joint_dataset, 
            epochs = epochs, steps_per_epoch = steps_per_epoch, validation_data = tf_joint_val_dataset, validation_steps = val_steps, verbose = 2, callbacks = [saver, tensorboard_callback])

    hist_df = pd.DataFrame(history.history)

    return model, hist_df
