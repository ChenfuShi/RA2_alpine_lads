import logging
import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from utils.config import Config
from dream.preprocessing import image_preprocessing, predict_joints
from dream.prediction import predict_dream_test_set

def _log_container_details():
    logging.info('Running version: %s', os.environ['CURR_VERSION'])
    logging.info('TF: %s', tf.__version__)
    logging.info('TF Addons: %s', tfa.__version__) 
    logging.info('GPU available: %s', tf.test.is_gpu_available())

def _preprocess_images(config):
    train_images, test_images = image_preprocessing(config)
    logging.info('Preprocessed %d training images', len(train_images))
    logging.info('Preprocessed %d test images', len(test_images))

    return train_images, test_images

def _predict_joints_in_images(config, train_images, test_images):
    predict_joints(config, train_images, test_images)

    # Sanity check joint CSV files
    for joint_file in ['dream_train_hand_joint_data.csv', 'dream_train_feet_joint_data.csv', 'dream_test_hand_joint_data.csv', 'dream_test_feet_joint_data.csv']:
        df = pd.read_csv('/output/' + joint_file)
        logging.info('Created %s joint dataframe in file %s', df.shape, joint_file)

def _predict_joint_damage(config):
    predict_params = {
        'hands_joint_source': '/output/dream_test_hand_joint_data.csv',
        'feet_joint_source': '/output/dream_test_feet_joint_data.csv',
        'hands_narrowing_model': '../resources/hands_narrowing_adam_no_weights_val.h5',
        'wrists_narrowing_model': '../resources/wrists_narrowing_adam_no_weights_val.h5',
        'feet_narrowing_model': '../resources/feet_narrowing_adam_no_weights_val.h5',
        'hands_erosion_model': '../resources/hands_erosion_adam_no_weights_val.h5',
        'wrists_erosion_model': '../resources/wrists_erosion_adam_no_weights_val.h5',
        'feet_erosion_model': '../resources/feet_erosion_adam_no_weights_val.h5',
        'template_path': '/test/template.csv',
        'output_path': '/output/predictions.csv',
    }

    predict_dream_test_set(config, predict_params)

def _clean_output():
    for file in os.listdir('/output'):
        if file != 'predictions.csv' and not file.endswith('.log'):
            file_path = '/output/' + file

            if os.path.isfile(file_path):
                os.remove(file_path)

# Change to dir
os.chdir('/usr/local/bin/ra_joint_predictions/')

config = Config('./utils/docker-config.json')

_log_container_details()
train_images, test_images = _preprocess_images(config)
_predict_joints_in_images(config, train_images, test_images)
_predict_joint_damage(config)

_clean_output()