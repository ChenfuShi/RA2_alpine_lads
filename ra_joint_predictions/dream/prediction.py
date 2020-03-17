import logging

import pandas as pd

from prediction.joint_damage import predict_test_set

def predict_dream_test_set(config):
    hands_joint_source = '/output/dream_test_hand_joint_data.csv'
    feet_joint_source = '/output/dream_test_feet_joint_data.csv'

    with open('./dream/dream_model_parameters_collection.json') as model_parameters_collection_file:
        model_parameters_collection = json.load(model_parameters_collection_file)

    logging.info('Running predictions using model setup: %s', model_parameters_collection)
        
    predictions_df = predict_test_set(config, model_parameters_collection, hands_joint_source = hands_joint_source, feet_joint_source = feet_joint_source)

    predictions_df.to_csv('/output/predictions.csv', index = False)