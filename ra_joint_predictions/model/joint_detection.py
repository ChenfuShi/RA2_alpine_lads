import tensorflow as tf
from tensorflow import keras
from model.utils.building_blocks_landmarks import bigger_kernel_base

def create_foot_joint_detector(config, path_to_weights = './weights/joint_detector_weights/feet/feet_joint_detector_weights'):
    # 6 Joints overall
    output_size = 12

    return _create_model_with_weigths(config, output_size, path_to_weights)

def create_hand_joint_detector(config, path_to_weights = './weights/joint_detector_weights/hands/hand_joint_detector_weights'):
    # 16 Joints overall
    output_size = 26

    return _create_model_with_weigths(config, output_size, path_to_weights)


# THESE TWO FUNCTIONS NEED TO USE THE LANDMARKS MODEL INSTEAD OF THEIR OWN BUILDING BLOCKS
def _create_model_with_weigths(config, output_size, path_to_weights):
    joints_detector_model = _create_joints_detector_model(config, output_size)

    joints_detector_model.load_weights(path_to_weights)

    return joints_detector_model

def _create_joints_detector_model(config, output_size):
    # larger kernel
    joints_detector_model = bigger_kernel_base(config)

    joints_detector_model.add(keras.layers.Dense(256, activation='relu'))
    joints_detector_model.add(keras.layers.BatchNormalization())
    joints_detector_model.add(keras.layers.Dense(output_size, activation='linear'))

    return joints_detector_model

