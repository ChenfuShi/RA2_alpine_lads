import os

from utils.config import Config

from model.joint_detection import create_hand_joint_detector
from prediction.joint_detection import rsna_joint_detector

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/RA2_alpine_lads/ra_joint_predictions')

    config = Config()

    hand_detector = create_hand_joint_detector(config)

    joint_detector = rsna_joint_detector(config, hand_detector)

    hand_dataframe = joint_detector.create_rnsa_dataset()

    hand_dataframe.to_csv('./data/predictions/rsna_joint_data.csv')
