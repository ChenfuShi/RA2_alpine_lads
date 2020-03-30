########################################

# takes pretrained weights with faces, then trains the on joints to predict landmarks



########################################

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

from utils.config import Config
from utils.dataset_preprocessing import fix_images
import model
import dataset
import logging
from model import landmarks_model
from train.train_landmarks import train_landmarks
from tensorflow.keras.models import load_model
import dataset.landmarks_dataset as ld_dataset
from tensorflow import keras

if __name__ == '__main__':

    # set up configuration
    configuration = Config()

    logging.info("doing preprocessing")
    fix_images(configuration)
    logging.info("preprocessing done")

    logging.info("starting of feet pre-training")
    joints_model = landmarks_model.resnet_landmarks_model(configuration,12,"weights/FACE_pretrain_resnet_model_250",10)
    joints_model.summary()

    landmark_dataset, landmark_dataset_val = ld_dataset.feet_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"FEET_pretrain_resnet")

    logging.info("starting of hands pre-training")
    joints_model = landmarks_model.resnet_landmarks_model(configuration,26,"weights/FACE_pretrain_resnet_model_250",10)
    joints_model.summary()

    landmark_dataset, landmark_dataset_val = ld_dataset.hands_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"HANDS_pretrain_resnet")

    logging.info("training on double pretrained model feet")

    joints_model = landmarks_model.resnet_landmarks_model(configuration,26,"weights/FEET_pretrain_resnet_model_2500",12)
    landmark_dataset, landmark_dataset_val = ld_dataset.hands_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)
    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"HANDS_train_resnet")

    logging.info("training on double pretrained model hands")

    joints_model = landmarks_model.resnet_landmarks_model(configuration,12,"weights/HANDS_pretrain_resnet_model_2500",26)
    landmark_dataset, landmark_dataset_val = ld_dataset.feet_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)
    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"FEET_train_resnet")
