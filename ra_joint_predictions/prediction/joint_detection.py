import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.ops.image_ops as img_ops
import dataset.ops.landmark_ops as lm_ops

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

        joint_dataframe = self._get_joint_predictions(dataset, joint_detectors, coord_mapping)

        return data_frame.merge(joint_dataframe)

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
    
        for image_name, landmark_image, original_image_shape in joint_prediction_dataset:
            img_name = image_name.numpy().decode('UTF-8')
            
            # Predict Landmark positions
            joint_predictions = self._cascading_joint_detection(landmark_image, img_name, joint_detectors)

            if np.count_nonzero(joint_predictions < 0) != 0:
                logging.error('No detector worked for image %s!', img_name)

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

        return pd.DataFrame(joint_predictions_list, index = np.arange(len(joint_predictions_list)))
    
    def _cascading_joint_detection(self, landmark_image, img_name, joint_detectors):
        for idx, joint_detector in enumerate(joint_detectors):
            joint_predictions = joint_detector.predict(landmark_image)[0]

            if np.count_nonzero(joint_predictions < 0) == 0:
                break
            else:
                logging.warn('Detector %d failed for image %s with landmarks less than 0', idx, img_name)
        
        return joint_predictions

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

        data_hands = self._detect_joints_in_image_data(hand_dataframe, img_dir, self.hand_joint_detectors, hand_coord_mapping)
        data_feet = self._detect_joints_in_image_data(feet_dataframe, img_dir, self.feet_joint_detectors, foot_coord_mapping)

        return data_hands, data_feet

class rsna_joint_detector(joint_detector):
    def __init__(self, config, hand_joint_detectors):
        super().__init__(config, consider_flip = False)

        self.hand_joint_detectors = hand_joint_detectors
        self.img_dir = config.rsna_img_dir
        
    def create_rnsa_dataset(self, img_dir = '../../rsna_boneAge/checked_rsna_training'):
        rsna_dataframe = self._create_detection_dataframe(img_dir)

        return self._detect_joints_in_image_data(rsna_dataframe, img_dir, self.hand_joint_detectors, hand_coord_mapping)