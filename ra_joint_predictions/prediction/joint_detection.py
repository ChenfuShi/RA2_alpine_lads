import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.ops.image_ops as img_ops
import dataset.ops.landmark_ops as lm_ops

from dataset.landmarks_dataset import _create_landmarks_dataframe
from utils.file_utils import is_supported_image_file

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Map each joint, to its indices in the predicted locations
hand_coord_mapping = {
    'mcp': [0, 2],
    'pip_2': [2, 4],
    'pip_3': [4, 6],
    'pip_4': [6, 8],
    'pip_5': [8, 10],
    'mcp_1': [10, 12],
    'mcp_2': [12, 14],
    'mcp_3': [14, 16],
    'mcp_4': [16, 18],
    'mcp_5': [18, 20],
    'w1': [20, 22],
    'w2': [22, 24],
    'w3': [24, 26]
}

foot_coord_mapping = {
    'mtp': [0, 2], 
    'mtp_1': [2, 4],
    'mtp_2': [4, 6],
    'mtp_3': [6, 8],
    'mtp_4': [8, 10],
    'mtp_5': [10, 12]
}

class joint_detector():
    def __init__(self, config, consider_flip = False):
        self.landmark_img_height = config.landmarks_img_height
        self.landmark_img_width = config.landmarks_img_width
        self.consider_flip = consider_flip

    # Go through each image in targe directory and create an entry in dataframe
    def _create_detection_dataframe(self, img_dir, file_names = None):
        if not file_names:
            file_names = os.listdir(img_dir)

        elements = []
        for file_name in file_names:
            file_parts = file_name.split('.')

            if(is_supported_image_file(file_parts)):
                flip = 'N'

                # Always False if consider_flip = False
                should_flip = self.consider_flip and '-R' in file_name
                if should_flip:
                    flip = 'Y'

                element = {
                    'image_name': file_parts[0],
                    'file_type': file_parts[1],
                    'flip': flip
                }

                elements.append(element)

        return pd.DataFrame(elements, index = np.arange(len(elements)))

    def _detect_joints_in_image_data(self, data_frame, img_dir, joint_detectors, coord_mapping):
        dataset = self._create_joint_detection_dataset(data_frame, img_dir)

        joint_dataframe, invalid_images = self._get_joint_predictions(dataset, joint_detectors, coord_mapping)

        return data_frame.merge(joint_dataframe), invalid_images

    def _create_joint_detection_dataset(self, data_frame, img_dir):
        image_info = data_frame[['image_name', 'file_type', 'flip']].values

        dataset = tf.data.Dataset.from_tensor_slices((image_info))
        dataset = self._map_to_landmark_detection_images(dataset, img_dir)
        dataset = dataset.prefetch(buffer_size = AUTOTUNE)

        return dataset

    def _map_to_landmark_detection_images(self, dataset, img_dir):
        def __load_joints(file_info):
            image, _ = img_ops.load_image(file_info, [], img_dir)
            landmark_detection_image, _ = img_ops.resize_image(image, [], self.landmark_img_height, self.landmark_img_width)

            return file_info[0], tf.expand_dims(landmark_detection_image, 0), tf.shape(image)

        return dataset.map(__load_joints, num_parallel_calls = AUTOTUNE)

    def _get_joint_predictions(self, joint_prediction_dataset, joint_detectors, coord_mapping):
        joint_predictions_list = []
        invalid_images = []
    
        for image_name, landmark_image, original_image_shape in joint_prediction_dataset:
            img_name = image_name.numpy().decode('UTF-8')
            
            # Predict Landmark positions
            joint_predictions, is_valid = self._cascading_joint_detection(landmark_image, img_name, joint_detectors)

            if is_valid is True:
                # Scale landmarks to original img size
                upscaled_joint_locations = lm_ops.upscale_detected_landmarks(joint_predictions, (self.landmark_img_height, self.landmark_img_width), original_image_shape)

                joint_prediction = {
                    'image_name': img_name
                }

                for key in coord_mapping:
                    coords = coord_mapping[key]
                    joint_coordinates = upscaled_joint_locations[coords[0]:coords[1]].numpy()

                    coord_x = joint_coordinates[0]
                    coord_y = joint_coordinates[1]

                    joint_prediction[key + '_x'] = coord_x
                    joint_prediction[key + '_y'] = coord_y

                joint_predictions_list.append(joint_prediction)
            else:
                logging.error('No detector worked for image %s!', img_name)
                
                invalid_images.append(img_name)

        return pd.DataFrame(joint_predictions_list, index = np.arange(len(joint_predictions_list))), invalid_images
    
    def _cascading_joint_detection(self, landmark_image, img_name, joint_detectors):
        for idx, joint_detector in enumerate(joint_detectors):
            img_shape = tf.shape(landmark_image)
            
            joint_predictions = joint_detector.predict(landmark_image)[0]
            
            is_valid = self._validate_joint_predictions(joint_predictions, img_shape, img_name, idx)
            
            if is_valid:
                break
        
        return joint_predictions, is_valid
    
    def _validate_joint_predictions(self, joint_predictions, img_shape, img_name, idx):
        is_valid = True
        
        if np.count_nonzero(joint_predictions < 0) != 0:
            logging.warn(f'Detector {idx} failed for image {img_name} with landmarks less than 0')
            
            is_valid = False
        elif (np.count_nonzero(joint_predictions[0::2] > img_shape[2]) > 0 or np.count_nonzero(joint_predictions[1::2] > img_shape[1]) > 0):
            logging.warn(f'Detector {idx} failed for image {img_name} with landmarks outside the image')
            
            is_valid = False
            
        return is_valid

class dream_joint_detector(joint_detector):
    def __init__(self, config, hand_joint_detectors, feet_joint_detectors):
        super().__init__(config, consider_flip = True)

        self.hand_joint_detectors = hand_joint_detectors
        self.feet_joint_detectors = feet_joint_detectors

    def create_dream_datasets(self, img_dir, file_names = None):
        full_dataframe = self._create_detection_dataframe(img_dir, file_names)

        hands_mask = ['H' in image_name for image_name in full_dataframe['image_name']]
        feet_mask = ['F' in image_name for image_name in full_dataframe['image_name']]

        hand_dataframe = full_dataframe.iloc[hands_mask]
        feet_dataframe = full_dataframe.iloc[feet_mask]

        data_hands, hands_invalid_images = self._detect_joints_in_image_data(hand_dataframe, img_dir, self.hand_joint_detectors, hand_coord_mapping)
        data_feet, feet_invalid_images = self._detect_joints_in_image_data(feet_dataframe, img_dir, self.feet_joint_detectors, foot_coord_mapping)

        return data_hands, data_feet, hands_invalid_images, feet_invalid_images

class rsna_joint_detector(joint_detector):
    def __init__(self, config, hand_joint_detectors):
        super().__init__(config, consider_flip = False)

        self.hand_joint_detectors = hand_joint_detectors
        self.img_dir = config.rsna_img_dir
        
    def create_rnsa_dataset(self, img_dir = '../../rsna_boneAge/checked_rsna_training'):
        rsna_dataframe = self._create_detection_dataframe(img_dir)

        return self._detect_joints_in_image_data(rsna_dataframe, img_dir, self.hand_joint_detectors, hand_coord_mapping)

def create_train_joint_dataframe(config, type, save_location):
    landmarks_location = os.path.join(config.landmarks_location, type)

    if type == 'hands':
        coord_mapping = hand_coord_mapping
        lm_columns = ['mcp_x', 'mcp_x', 'pip_2_x', 'pip_2_y', 'pip_3_x', 'pip_3_y', 'pip_4_x', 'pip_4_y', 'pip_5_x', 'pip_5_y', 'mcp_1_x', 'mcp_1_y', 'mcp_2_x', 'mcp_2_y', 'mcp_3_x', 'mcp_3_y', 'mcp_4_x', 'mcp_4_y', 'mcp_5_x', 'mcp_5_y', 'w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']
    else:
        coord_mapping = foot_coord_mapping
        lm_columns = ['mtp_x', 'mtp_y', 'mtp_1_x', 'mtp_1_y', 'mtp_2_x', 'mtp_2_y', 'mtp_3_x', 'mtp_3_y', 'mtp_4_x', 'mtp_4_y', 'mtp_5_x', 'mtp_5_y']

    df = _create_landmarks_dataframe(landmarks_location)

    joint_predictions_list = []

    for _, row in df.iterrows():
        flip = row['flip']
        y = row[lm_columns].to_numpy()

        if flip:
            file_info = row[['sample_id', 'file_type', 'flip']]

            img, _ = img_ops.load_image(file_info, [], config.train_fixed_location)

            y = tf.Variable(y, dtype = tf.float64)

            y = lm_ops.flip_landmarks(y, img.shape)

        joint_prediction = {
            "image_name": file_info[0],
            "file_type": file_info[1],
            "flip": file_info[2]
        }

        for key in coord_mapping:
            coords = coord_mapping[key]
            joint_coordinates = y[coords[0]:coords[1]].numpy()

            coord_x = joint_coordinates[0]
            coord_y = joint_coordinates[1]

            joint_prediction[key + '_x'] = coord_x
            joint_prediction[key + '_y'] = coord_y

        joint_predictions_list.append(joint_prediction)

    joint_predictions_df = pd.DataFrame(joint_predictions_list, index = np.arange(len(joint_predictions_list)))
    joint_predictions_df.to_csv(save_location, index = False)