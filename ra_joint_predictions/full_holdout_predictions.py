import logging
import os
import sys

from utils.config import Config
from utils.saver import save_pretrained_model

from dream.prediction import predict_dream_test_set

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    config = Config()

    predict_params = {
        'hands_joint_source': './data/predictions/hand_joint_data_test.csv',
        'feet_joint_source': './data/predictions/feet_joint_data_test.csv',
        'hands_narrowing_model': '../trained_models/v2/hands_narrowing_v2_val.h5',
        'wrists_narrowing_model': '../trained_models/v2/wrists_narrowing_v2_val.h5',
        'feet_narrowing_model': '../trained_models/v2/feet_narrowing_v2_val.h5',
        'hands_erosion_model': '../trained_models/v2/hands_erosion_v2_val.h5',
        'wrists_erosion_model': '../trained_models/v2/wrists_erosion_v2_val.h5',
        'feet_erosion_model': '../trained_models/v2/feet_erosion_v2_val.h5',
        'template_path': '../resources/template.csv',
        'output_path': '../trained_models/output.csv',
    }

    predict_dream_test_set(config, predict_params)
