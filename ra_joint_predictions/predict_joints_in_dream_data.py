import os

from tensorflow.keras.models import load_model

from utils.config import Config

from model.joint_detection import create_foot_joint_detector, create_hand_joint_detector
from prediction.joint_detection import dream_joint_detector

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    config = Config()

    hand_detector_model = load_model("../resources/hands_landmarks_original_epoch_1000_predictor_1.h5")
    foot_detector_model = load_model("../resources/feet_landmarks_original_epoch_1000_predictor_1.h5")

    joint_detector = dream_joint_detector(config, hand_detector_model, foot_detector_model)

    hand_dataframe, feet_dataframe = joint_detector.create_dream_datasets(config.train_fixed_location)

    hand_dataframe.to_csv('./data/predictions/hand_joint_data_v2.csv')
    feet_dataframe.to_csv('./data/predictions/feet_joint_data_v2.csv')
