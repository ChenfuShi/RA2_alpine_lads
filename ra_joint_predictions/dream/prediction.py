import json
import logging
import pandas as pd
import pickle

from prediction.joint_damage import predict_test_set

def predict_dream_test_set(config):
    hands_joint_source = '/output/dream_test_hand_joint_data.csv'
    feet_joint_source = '/output/dream_test_feet_joint_data.csv'

    test_hands_invalid_images = pickle.load(open("/output/test_hands_invalid_images.data", "rb"))
    test_feet_invalid_images = pickle.load(open("/output/test_feet_invalid_images.data", "rb"))

    with open('./dream/dream_model_parameters_collection.json') as model_parameters_collection_file:
        model_parameters_collection = json.load(model_parameters_collection_file)

    logging.info('Running predictions using model setup: %s', model_parameters_collection)
        
    predictions_df = predict_test_set(config, model_parameters_collection, hands_joint_source = hands_joint_source, feet_joint_source = feet_joint_source, hands_invalid_images = test_hands_invalid_images, feet_invalid_images = test_feet_invalid_images)

    predictions_df.to_csv('/output/predictions.csv', index = False)