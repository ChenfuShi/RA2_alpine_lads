import logging
import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from utils.config import Config
from model.joint_detection import create_hand_joint_detector
from prediction.joint_detection import rsna_joint_detector


config = Config()

hand_detector_model = load_model("./resources/hands_landmarks_original_epoch_1000_predictor_1.h5")

joint_detector = rsna_joint_detector(config, hand_detector_model)

hand_dataframe = joint_detector.create_rnsa_dataset()

hand_dataframe.to_csv('./data/predictions/rsna_joint_data.csv')
