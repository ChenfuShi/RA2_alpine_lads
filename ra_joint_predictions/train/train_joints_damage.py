import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Input

from dataset.joint_dataset import feet_joint_dataset, hands_joints_dataset, hands_wrists_dataset
from model.utils.metrics import top_2_categorical_accuracy
from model.utils.metrics import argmax_rsme, softmax_rsme_metric, class_softmax_rsme_metric
from utils.saver import CustomSaver, _get_tensorboard_callback

def train_joints_damage_model(config, model_name, pretrained_base_model, joint_type, dmg_type):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)
    
    joint_dataset, tf_joint_dataset = _get_dataset(config, joint_type, dmg_type)
    
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

    # history = model.fit(
        #tf_joint_dataset, epochs = 100, steps_per_epoch = 75, verbose = 2, class_weight = joint_dataset.class_weights[0], callbacks = [saver, tensorboard_callback]
    #)

    #hist_df = pd.DataFrame(history.history)
    #hist_df.to_csv('./' + model_name + '_hist.csv')

    return model

def _get_dataset(config, joint_type, dmg_type):
    outcomes_sources = os.path.join(config.train_location, 'training.csv')

    erosion_flag = dmg_type == 'E'

    if joint_type == 'F':
        joint_dataset = feet_joint_dataset(config)

        tf_joint_dataset = joint_dataset.create_feet_joints_dataset(outcomes_sources, joints_source = './data/predictions/feet_joint_data.csv', erosion_flag = erosion_flag)
    elif joint_type == 'H':
        joint_dataset = hands_joints_dataset(config)

        tf_joint_dataset = joint_dataset.create_hands_joints_dataset(outcomes_sources, joints_source = './data/predictions/hand_joint_data.csv', erosion_flag = erosion_flag)
    else:
        joint_dataset = hands_wrists_dataset(config)

        tf_joint_dataset = joint_dataset.create_wrists_joints_dataset(outcomes_sources, joints_source = './data/predictions/hand_joint_data.csv', erosion_flag = erosion_flag)

    return joint_dataset, tf_joint_dataset
