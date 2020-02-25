import logging
import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from utils.config import Config
from dream import execute_dream_predictions
from dream.preprocessing import image_preprocessing, predict_joints

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

def _clean_output():
    for file in os.listdir('/output'):
        if file != 'predictions.csv' and not file.endswith('.log'):
            os.remove('/output/' + file)

# Change to dir
os.chdir('/usr/local/bin/ra_joint_predictions/')

config = Config('./utils/docker-config.json')

_log_container_details()
train_images, test_images = _preprocess_images(config)
_predict_joints_in_images(config, train_images, test_images)

execute_dream_predictions()

_clean_output()