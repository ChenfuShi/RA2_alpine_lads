import datetime
import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense

from dataset.joint_dataset import feet_joint_dataset
from model.utils.metrics import top_2_categorical_accuracy
from model.utils.metrics import argmax_rsme, softmax_rsme_metric, class_softmax_rsme_metric
from utils.saver import CustomSaver, _get_tensorboard_callback

def train_feet_erosion_model(config, model_name, pretrained_base_model):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)
    
    dataset = feet_joint_dataset(config)
    
    outcomes_sources = os.path.join(config.train_location, 'training.csv')
    
    feet_joint_erosion_dataset, val_dataset = dataset.create_feet_joints_dataset(outcomes_sources, joints_source = './data/predictions/feet_joint_data_train.csv', val_joints_source = './data/predictions/feet_joint_data_test.csv')

    output_bias = tf.keras.initializers.Constant(dataset.class_bias[0])
    
    pretrained_base_model.add(Dense(5, activation = 'softmax', name = 'main_output', bias_initializer = output_bias))

    metrics = ['categorical_accuracy', softmax_rsme_metric(np.arange(5)), argmax_rsme, class_softmax_rsme_metric(np.arange(5), 0)]
    pretrained_base_model.compile(loss = 'categorical_crossentropy', metrics = metrics, optimizer = 'adam')

    history = pretrained_base_model.fit(
        feet_joint_erosion_dataset, epochs = 500, steps_per_epoch = 75, validation_data = val_dataset, 
        validation_steps = 15, verbose = 2, class_weight = dataset.class_weights[0], callbacks = [saver, tensorboard_callback]
    )

    return pretrained_base_model