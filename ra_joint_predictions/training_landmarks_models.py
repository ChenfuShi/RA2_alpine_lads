########################################

# takes pretrained weights with faces, then trains the on joints to predict landmarks



########################################

from utils.config import Config
from utils.dataset_preprocessing import fix_images
import model
import dataset
import logging
from model import landmarks_model
from train.train_landmarks import train_landmarks
from tensorflow.keras.models import load_model
import dataset.landmarks_dataset as ld_dataset
import dataset.landmarks_faces_pretrain_dataset as faces_dataset
from train.pretrain_faces import pretrain_faces
from tensorflow import keras

if __name__ == '__main__':

    # set up configuration
    configuration = Config()


    dataset = faces_dataset.landmark_pretrain_faces_dataset(configuration)
    faces, faces_val = dataset.create_train_dataset()

    logging.info("datasets prepared")

    # define model
    model = landmarks_model.basic_landmarks_model(configuration,10)
    model.summary()

    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_faces(model,faces,faces_val,configuration,"FACE_original_retrained", epochs = 101)
    

    logging.info("doing preprocessing")
    fix_images(configuration)
    logging.info("preprocessing done")

    logging.info("starting feet pre-training")
    joints_model = landmarks_model.basic_landmarks_model(configuration,12,"weights/FACE_original_retrained_model_100",10)
    joints_model.summary()

    landmark_dataset, landmark_dataset_val = ld_dataset.feet_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"FEET_pretrain_original",epochs=1001)

    logging.info("starting hands pre-training")
    joints_model = landmarks_model.basic_landmarks_model(configuration,26,"weights/FACE_original_retrained_model_100",10)
    joints_model.summary()

    landmark_dataset, landmark_dataset_val = ld_dataset.hands_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"HANDS_pretrain_original",epochs=1001)

    logging.info("training on double pretrained model feet")

    joints_model = landmarks_model.basic_landmarks_model(configuration,26,"weights/FEET_pretrain_original_model_1000",12)
    landmark_dataset, landmark_dataset_val = ld_dataset.hands_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)
    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"HANDS_train_original",epochs=1001)

    logging.info("training on double pretrained model hands")

    joints_model = landmarks_model.basic_landmarks_model(configuration,12,"weights/HANDS_pretrain_original_model_1000",26)
    landmark_dataset, landmark_dataset_val = ld_dataset.feet_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)
    train_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"FEET_train_original",epochs=1001)
    
