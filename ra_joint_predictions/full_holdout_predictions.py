import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

import json
import logging
import os
import sys

import pandas as pd

from utils.config import Config
from utils.saver import save_pretrained_model

from prediction.joint_damage import predict_test_set

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    config = Config()

    output_file = sys.argv[1]

    with open('./dream/dream_model_parameters_collection.json') as model_parameters_collection_file:
        model_parameters_collection = json.load(model_parameters_collection_file)

    logging.info(f"Running full holdout predictions, with params {model_parameters_collection}, writing to output file: {output_file}")

    hands_joint_source = './data/predictions/hand_joint_data_test_v2.csv'
    hands_df = pd.read_csv(hands_joint_source)
    
    hands_invalid_images = hands_df['image_name'].to_numpy()
    
    predictions = predict_test_set(config, model_parameters_collection, hands_invalid_images = hands_invalid_images)
    predictions.to_csv(f'../trained_models/{output_file}.csv', index = False)