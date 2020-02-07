########################################

# this piece of junk is such a mess


########################################

from utils.config import Config
from dataset.dataset_preprocessing import fix_images
import model
import dataset
import logging
from model import landmarks_model
from utils.pretrain import pretrain_faces,pretrain_landmarks
from tensorflow.keras.models import load_model
import dataset.landmarks_dataset as ld_dataset
import dataset.landmarks_pretrain_dataset as land_pre
from tensorflow import keras

if __name__ == '__main__':

    # set up configuration
    configuration = Config()
    # after this logging.info() will print save to a specific file (global)!
    # logging.info("configuration loaded")
    
    # # prepare data
    # logging.info("preparing train dataset faces")

    # dataset = land_pre.landmark_pretrain_faces_dataset(configuration)
    # faces, faces_val = dataset.create_train_dataset()

    # logging.info("datasets prepared")

    # # define model
    # model = landmarks_model.landmarks_model_pretrain(configuration)
    # model.summary()

    # logging.info("model prepared")
    # # train
    # logging.info("starting training")
    # pretrain_faces(model,faces,faces_val,configuration,"FACE_LANDMARK_big_kernel_long")
    
    # logging.info("doing preprocessing")
    # fix_images(configuration)
    # logging.info("preprocessing done")

    # logging.info("starting step 2 of feet training")
    joints_model = landmarks_model.landmarks_model_feet(configuration,"weights/FACE_LANDMARK_big_kernel_long_model_70")
    joints_model.summary()

    landmark_dataset, landmark_dataset_val = ld_dataset.feet_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    pretrain_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"FEET_LANDMARK_big_kernel_long")


    # logging.info("starting step 2 of feet training")
    # joints_model = landmarks_model.landmarks_model_hands(configuration,"weights/FACE_LANDMARK_big_kernel_long_model_70")
    # joints_model.summary()

    # landmark_dataset, landmark_dataset_val = ld_dataset.hands_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    # pretrain_landmarks(joints_model,landmark_dataset,landmark_dataset_val,configuration,"HANDS_LANDMARK_big_kernel_long")


    # fuck it i'm hacking it 

    # landmark_dataset, landmark_dataset_val = ld_dataset.feet_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    # joints_model = landmarks_model.landmarks_model_hands(configuration)
    # joints_model.load_weights("weights/HANDS_LANDMARK_big_kernel_long_model_480")
    # model=keras.models.Sequential()
    # # remove last layer from pretrain model
    # for layer in joints_model.layers[:-1]:
    #     model.add(layer)
    # # add the layer back
    # model.add(keras.layers.Dense(12, activation='linear'))
    # model.compile(optimizer='adam',
    #         loss='mean_squared_error',
    #         metrics=['mae'])
    # pretrain_landmarks(model,landmark_dataset,landmark_dataset_val,configuration,"FEET_LANDMARK_test_with_hand_pretrain")


    landmark_dataset, landmark_dataset_val = ld_dataset.hands_landmarks_dataset(configuration).create_landmarks_dataset(create_val=True)

    joints_model = landmarks_model.landmarks_model_feet(configuration)
    joints_model.load_weights("weights/FEET_LANDMARK_big_kernel_long_model_495")
    model=keras.models.Sequential()
    # remove last layer from pretrain model
    for layer in joints_model.layers[:-1]:
        model.add(layer)
    # add the layer back
    model.add(keras.layers.Dense(26, activation='linear'))
    model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae'])
    pretrain_landmarks(model,landmark_dataset,landmark_dataset_val,configuration,"HANDS_LANDMARK_test_with_feet_pretrain")