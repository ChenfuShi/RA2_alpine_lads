import logging
import os
import sys

from utils.config import Config
from utils.saver import save_pretrained_model

from prediction.joint_damage import predict_test_set

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    config = Config()

    model_parameters_collection = {
        'hands_narrowing_model': { 'model': '../trained_models/adam_no_weights_reg/hands_narrowing_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'wrists_narrowing_model': { 'model': '../trained_models/adam_no_weights_reg/wrists_narrowing_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'feet_narrowing_model': { 'model': '../trained_models/adam_no_weights_reg/feet_narrowing_adam_no_weights_reg_shuffle_hand_pretrain.h5', 'is_regression': True },
        'hands_erosion_model': { 'model': '../trained_models/adam_no_weights_reg/hands_erosion_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'wrists_erosion_model': { 'model': '../trained_models/adam_no_weights_reg/wrists_erosion_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'feet_erosion_model': { 'model': '../trained_models/adam_no_weights_reg/feet_erosion_adam_no_weights_reg_shuffle_hand_pretrain.h5', 'is_regression': True }
    }

    predict_test_set(config, model_parameters_collection)
