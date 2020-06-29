import logging
import os
import pickle
import time

from prediction.joint_detection import dream_joint_detector
from tensorflow.keras.models import load_model
from utils.image_preprocessor import preprocess_images

def image_preprocessing(config):
    logging.info('Preprocessing images')

    start = time.time()

    preprocessed_training_images = preprocess_images(config.train_location, config.train_fixed_location)
    preprocessed_test_images = preprocess_images(config.test_location, config.test_fixed_location)

    end = time.time()
    logging.info('Preprocessed images in: %ds', end - start)

    return preprocessed_training_images, preprocessed_test_images

def predict_joints(config, train_images, test_images):
    logging.info('Predicting joints in images')

    start = time.time()
    
    hand_detector_model_v2 = load_model("../resources/hands_landmarks_original_epoch_1000_predictor_1.h5")
    foot_detector_model_v2 = load_model("../resources/feet_landmarks_original_epoch_1000_predictor_1.h5")

    hand_detector_model = load_model("../resources/hand_joint_detector_v1.h5")
    foot_detector_model = load_model("../resources/foot_joint_detector_v1.h5")

    joint_detector = dream_joint_detector(config, [hand_detector_model_v2, hand_detector_model], [foot_detector_model_v2, foot_detector_model])
    
    train_hand_dataframe, train_feet_dataframe, train_hands_invalid_images, train_feet_invalid_images = joint_detector.create_dream_datasets(config.train_fixed_location, train_images)
    test_hand_dataframe, test_feet_dataframe, test_hands_invalid_images, test_feet_invalid_images = joint_detector.create_dream_datasets(config.test_fixed_location, test_images)

    # Resave to different default here since we don't want to use pre-existing predictions, but use new ones
    train_hand_dataframe.to_csv('/output/dream_train_hand_joint_data.csv')
    train_feet_dataframe.to_csv('/output/dream_train_feet_joint_data.csv')

    pickle.dump(train_hands_invalid_images, open("/output/train_hands_invalid_images.data", "wb"))
    pickle.dump(train_feet_invalid_images, open("/output/train_feet_invalid_images.data", "wb"))

    test_hand_dataframe.to_csv('/output/dream_test_hand_joint_data.csv')
    test_feet_dataframe.to_csv('/output/dream_test_feet_joint_data.csv')

    pickle.dump(test_hands_invalid_images, open("/output/test_hands_invalid_images.data", "wb"))
    pickle.dump(test_feet_invalid_images, open("/output/test_feet_invalid_images.data", "wb"))

    end = time.time()
    logging.info('Predicted joints in: %ds', end - start)