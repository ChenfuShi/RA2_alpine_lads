import tensorflow as tf
from tensorflow import keras

def create_foot_joint_detector(config, path_to_weights = './weights/joint_detector_weights/feet/feet_joint_detector_weights'):
    # 6 Joints overall
    output_size = 12

    return _create_model_with_weigths(config, output_size, path_to_weights)

def create_hand_joint_detector(config, path_to_weights = './weights/joint_detector_weights/hands/hand_joint_detector_weights'):
    # 16 Joints overall
    output_size = 26

    return _create_model_with_weigths(config, output_size, path_to_weights)

def _create_model_with_weigths(config, output_size, path_to_weights):
    joints_detector_model = _create_joints_detector_model(config, output_size)

    joints_detector_model.load_weights(path_to_weights)

    return joints_detector_model

def _create_joints_detector_model(config, output_size):
    # larger kernel
    joints_detector_model = _create_model_kernel(config.landmarks_img_height, config.landmarks_img_width)

    joints_detector_model.add(keras.layers.Dense(256, activation='relu'))
    joints_detector_model.add(keras.layers.BatchNormalization())
    joints_detector_model.add(keras.layers.Dense(output_size, activation='linear'))

    return joints_detector_model

def _create_model_kernel(input_height, input_width):
    model_kernel = keras.models.Sequential([
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu",input_shape=[input_height, input_width, 1]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten()
    ])

    return model_kernel