import logging

import pandas as pd

from prediction.joint_damage import predict_test_set

def predict_dream_test_set(config):
    hands_joint_source = '/output/dream_test_hand_joint_data.csv'
    feet_joint_source = '/output/dream_test_feet_joint_data.csv'

    model_parameters_collection = {
        'hands_narrowing_model': { 'model': '../resources/hands_narrowing_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'wrists_narrowing_model': { 'model': '../resources/wrists_narrowing_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'feet_narrowing_model': { 'model': '../resources/feet_narrowing_adam_no_weights_reg_shuffle_hand_pretrain.h5', 'is_regression': True },
        'hands_erosion_model': { 'model': '../resources/hands_erosion_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'wrists_erosion_model': { 'model': '../resources/wrists_erosion_adam_no_weights_reg_shuffle.h5', 'is_regression': True },
        'feet_erosion_model': { 'model': '../resources/feet_erosion_adam_no_weights_reg_shuffle_hand_pretrain.h5', 'is_regression': True }
    }

    predictions_df = predict_test_set(config, model_parameters_collection, hands_joint_source = hands_joint_source, feet_joint_source = feet_joint_source)

    predictions_df.to_csv('/output/predictions.csv', index = False)