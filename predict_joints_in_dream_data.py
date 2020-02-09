from utils.config import Config

from model.joint_detection import create_foot_joint_detector, create_hand_joint_detector
from prediction.joints.joint_predictions import dream_joint_detector

if __name__ == '__main__':
    config = Config()

    hand_detector = create_hand_joint_detector(config)
    foot_detector = create_foot_joint_detector(config)

    joint_detector = dream_joint_detector(config, hand_detector, foot_detector)

    hand_dataframe, feet_dataframe = joint_detector.create_dream_dataframes()

    hand_dataframe.to_csv('./data/hand_joint_data.csv')
    feet_dataframe.to_csv('./data/feet_joint_data.csv')