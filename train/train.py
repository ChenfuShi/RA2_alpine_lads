import datetime

import tensorflow as tf

from tensorflow.keras.layers import Dense

from utils.saver import CustomSaver

from dataset.joint_dataset import feet_joint_dataset
from utils import top_2_categorical_accuracy

def train_feet_erosion_model(config, model_name, pretrained_base_model):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)
    
    dataset = feet_joint_dataset(config)
    feet_joint_erosion_dataset, val_dataset = dataset.create_feet_joints_dataset(narrowing_flag = True, joint_source = './data/feet_joint_data_train.csv', val_joints_source = './data/feet_joint_data_test.csv')

    output_bias = tf.keras.initializers.Constant(dataset.class_bias)
    # Add erosion outcomes to pretrained model
    pretrained_base_model.add(Dense(5, activation = 'softmax', name = 'main_output', bias_initializer = output_bias))

    pretrained_base_model.compile(loss = 'categorical_crossentropy', metrics=["categorical_accuracy", top_2_categorical_accuracy], optimizer='adam')

    history = pretrained_base_model.fit(
        feet_joint_erosion_dataset, epochs = 2000, steps_per_epoch = 75, validation_data = val_dataset, 
        validation_steps = 15, verbose = 2, class_weight = dataset.class_weights, callbacks = [saver, tensorboard_callback]
    )

    return pretrained_base_model


def _get_tensorboard_callback(model_name):
    log_dir = 'logs/tensorboard/' + model_name + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    return tensorboard_callback
