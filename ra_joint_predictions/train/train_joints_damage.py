import datetime
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Input

from dataset.joint_dataset import feet_joint_dataset, hands_joints_dataset, hands_wrists_dataset, joint_narrowing_dataset
from model.utils.metrics import top_2_categorical_accuracy
from model.utils.metrics import argmax_rsme, softmax_rsme_metric, class_softmax_rsme_metric
from utils.saver import CustomSaver, _get_tensorboard_callback

def train_joints_damage_model(config, model_name, pretrained_base_model, joint_type, dmg_type, do_validation = False):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)
    
    joint_dataset, tf_joint_dataset = _get_dataset(config, joint_type, dmg_type, do_validation = do_validation)
    
    metric_dir = {}
    outputs = []
    for idx, class_weight in enumerate(joint_dataset.class_weights):
        no_outcomes = len(class_weight.keys())
        
        metrics = ['categorical_crossentropy', softmax_rsme_metric(np.arange(no_outcomes)), argmax_rsme, class_softmax_rsme_metric(np.arange(no_outcomes), 0)]
        
        output = Dense(no_outcomes, activation = 'softmax', name = f'output_{idx}')(pretrained_base_model.output)
        outputs.append(output)
        
        metric_dir[f'output_{idx}'] = metrics

    model = keras.models.Model(
        inputs = pretrained_base_model.input,
        outputs = outputs,
        name = f'joint_model_{joint_type}_{dmg_type}')

    # metrics = ['categorical_accuracy', softmax_rsme_metric(np.arange(5)), argmax_rsme, class_softmax_rsme_metric(np.arange(5), 0)]
    model.compile(loss = 'categorical_crossentropy', metrics = metric_dir, optimizer = 'adam')

    if not do_validation:
        history = model.fit(
            tf_joint_dataset, epochs = 250, steps_per_epoch = 150, verbose = 2, class_weight = joint_dataset.class_weights[0], callbacks = [saver, tensorboard_callback])
    else:
        history = model.fit(
            tf_joint_dataset[0], epochs = 250, steps_per_epoch = 150, validation_data = tf_joint_dataset[1], validation_steps = 35,
                verbose = 2, class_weight = joint_dataset.class_weights[0], callbacks = [saver, tensorboard_callback])

    hist_df = pd.DataFrame(history.history)

    return model, hist_df

def _get_dataset(config, joint_type, dmg_type, do_validation = False):
    outcomes_sources = os.path.join(config.train_location, 'training.csv')

    if do_validation:
        hand_joints_source = './data/predictions/hand_joint_data_train.csv'
        feet_joints_source = './data/predictions/feet_joint_data_train.csv'
        hand_joints_val_source = './data/predictions/hand_joint_data_test.csv'
        feet_joints_val_source = './data/predictions/feet_joint_data_test.csv'
    else:
        hand_joints_source = './data/predictions/hand_joint_data.csv'
        feet_joints_source = './data/predictions/feet_joint_data.csv'
        hand_joints_val_source = None
        feet_joints_val_source = None

    erosion_flag = dmg_type == 'E'
    
    if joint_type == 'F':
        joint_dataset = feet_joint_dataset(config)

        return joint_dataset, joint_dataset.create_feet_joints_dataset(outcomes_sources, joints_source = feet_joints_source, val_joints_source = feet_joints_val_source, erosion_flag = erosion_flag)
    elif joint_type == 'H':
        joint_dataset = hands_joints_dataset(config)

        return joint_dataset, joint_dataset.create_hands_joints_dataset(outcomes_sources, joints_source = hand_joints_source, val_joints_source = hand_joints_val_source, erosion_flag = erosion_flag)
    elif joint_type == 'W':
        joint_dataset = hands_wrists_dataset(config)

        return joint_dataset, joint_dataset.create_wrists_joints_dataset(outcomes_sources, joints_source = hand_joints_source, val_joints_source = hand_joints_val_source, erosion_flag = erosion_flag)
    elif joint_type == 'HF' and not erosion_flag:
        joint_dataset = joint_narrowing_dataset(config)

        return joint_dataset, joint_dataset.create_combined_narrowing_joint_dataset(outcomes_sources, hand_joints_source = hand_joints_source, feet_joints_source = feet_joints_source, hand_joints_val_source = hand_joints_val_source, feet_joints_val_source = feet_joints_val_source)