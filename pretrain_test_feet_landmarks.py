########################################




########################################

from utils.config import Config
from dataset.dataset_preprocessing import fix_images
import model
import dataset
import logging
from model import landmarks_model
from utils.pretrain import pretrain_faces,pretrain_landmarks_no_val
from tensorflow.keras.models import load_model
import dataset.landmarks_dataset as ld_dataset
import dataset.landmarks_pretrain_dataset as land_pre

if __name__ == '__main__':

    # set up configuration
    configuration = Config()
    # after this logging.info() will print save to a specific file (global)!
    logging.info("configuration loaded")
    
    # prepare data
    logging.info("preparing train dataset faces")

    dataset = land_pre.landmark_pretrain_faces_dataset(configuration)
    faces, faces_val = dataset.create_train_dataset()

    logging.info("datasets prepared")

    # define model
    model = landmarks_model.landmarks_model_pretrain(configuration)
    model.summary()

    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_faces(model,faces,faces_val,configuration,"FACE_LANDMARK_big_kernel_long")


    logging.info("starting step 2 of feet training")
    joints_model = landmarks_model.landmarks_model_feet(configuration,"weights/FACE_LANDMARK_big_kernel_long_model_50")
    joints_model.summary()

    landmark_dataset = ld_dataset.landmarks_dataset(configuration).create_landmarks_dataset()

    pretrain_landmarks_no_val(joints_model,landmark_dataset,configuration,"FEET_LANDMARK_big_kernel_long")