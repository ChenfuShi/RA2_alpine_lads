import os

from utils.config import Config

from tensorflow.keras.models import load_model
from model.joint_detection import create_hand_joint_detector
from prediction.joint_detection import rsna_joint_detector

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/RA2_alpine_lads/ra_joint_predictions')

    config = Config()
    
    hand_detector_model_v2 = load_model("./resources/hands_landmarks_original_epoch_1000_predictor_1.h5")
    hand_detector_model = load_model("../resources/hand_joint_detector_v1.h5")

    joint_detector = rsna_joint_detector(config, [hand_detector_model_v2, hand_detector_model])

    hand_dataframe = joint_detector.create_rnsa_dataset()
    hand_dataframe.to_csv('./data/predictions/rsna_joint_data.csv')