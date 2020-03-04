import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense

from dataset.joint_dataset import feet_joint_dataset
from model.utils.metrics import top_2_categorical_accuracy
from model.utils.metrics import argmax_rsme, softmax_rsme_metric, class_softmax_rsme_metric
from utils.saver import CustomSaver, _get_tensorboard_callback

def train_feet_narrowing_model(config, model_name, pretrained_base_model):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)
    
    dataset = feet_joint_dataset(config)
    
    outcomes_sources = os.path.join(config.train_location, 'training.csv')
    
    feet_joint_erosion_dataset = dataset.create_feet_joints_dataset(outcomes_sources, joints_source = './data/predictions/feet_joint_data.csv')

    # output_bias = tf.keras.initializers.Constant(dataset.class_bias[0])
    
    pretrained_base_model.add(Dense(5, activation = 'softmax', name = 'main_output')) # , bias_initializer = output_bias))

    metrics = ['categorical_accuracy', softmax_rsme_metric(np.arange(5)), argmax_rsme, class_softmax_rsme_metric(np.arange(5), 0)]
    pretrained_base_model.compile(loss = 'categorical_crossentropy', metrics = metrics, optimizer = 'adam')

    history = pretrained_base_model.fit(
        feet_joint_erosion_dataset, epochs = 100, steps_per_epoch = 75, verbose = 2, class_weight = dataset.class_weights[0], callbacks = [saver, tensorboard_callback]
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('./' + model_name + '_hist.csv')

    return pretrained_base_model
