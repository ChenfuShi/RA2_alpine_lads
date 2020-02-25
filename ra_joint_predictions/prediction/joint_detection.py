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

# Map each joint, to its outcome scores
hand_outcome_mapping = {
    'mcp': [[], ['{part}_mcp_E__ip'] ],
    'pip_2': [['{part}_pip_J__2'], ['{part}_pip_E__2']],
    'pip_3': [['{part}_pip_J__3'], ['{part}_pip_E__3']],
    'pip_4': [['{part}_pip_J__4'], ['{part}_pip_E__4']],
    'pip_5': [['{part}_pip_J__5'], ['{part}_pip_E__5']],
    'mcp_1': [['{part}_mcp_J__1'], ['{part}_mcp_E__1']],
    'mcp_2': [['{part}_mcp_J__2'], ['{part}_mcp_E__2']],
    'mcp_3': [['{part}_mcp_J__3'], ['{part}_mcp_E__3']],
    'mcp_4': [['{part}_mcp_J__4'], ['{part}_mcp_E__4']],
    'mcp_5': [['{part}_mcp_J__5'], ['{part}_mcp_E__5']],
    'w1': [['{part}_wrist_J__radcar', '{part}_wrist_J__mna'], ['{part}_wrist_E__mc1', '{part}_wrist_E__mul']],
    'w2': [['{part}_wrist_J__cmc3', '{part}_wrist_J__capnlun'], ['{part}_wrist_E__radius', '{part}_wrist_E__nav']],
    'w3': [['{part}_wrist_J__cmc4', '{part}_wrist_J__cmc5'], ['{part}_wrist_E__ulna', '{part}_wrist_E__lunate']]
}

foot_outcome_mapping = {
    'mtp': [['{part}_mtp_J__ip'], ['{part}_mtp_E__ip']], 
    'mtp_1': [['{part}_mtp_J__1'], ['{part}_mtp_E__1']], 
    'mtp_2': [['{part}_mtp_J__2'], ['{part}_mtp_E__2']],
    'mtp_3': [['{part}_mtp_J__3'], ['{part}_mtp_E__3']],
    'mtp_4': [['{part}_mtp_J__4'], ['{part}_mtp_E__4']],
    'mtp_5': [['{part}_mtp_J__5'], ['{part}_mtp_E__5']]
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

    def _detect_joints_in_image_data(self, data_frame, img_dir, joint_detector, coord_mapping):
        dataset = self._create_joint_detection_dataset(data_frame, img_dir)

        joint_dataframe = self._get_joint_predictions(dataset, joint_detector, coord_mapping)

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

    def _get_joint_predictions(self, joint_prediction_dataset, joint_detector, coord_mapping):
        joint_predictions_list = []
    
        for image_name, landmark_image, original_image_shape in joint_prediction_dataset:
            # Predict Landmark positions
            joint_predictions = joint_detector.predict(landmark_image)[0]

            # Scale landmarks to original img size
            upscaled_joint_locations = lm_ops.upscale_detected_landmarks(joint_predictions, (self.landmark_img_height, self.landmark_img_width), original_image_shape)

            joint_prediction = {
                'image_name': image_name.numpy().decode('UTF-8')
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

class dream_joint_detector(joint_detector):
    def __init__(self, config, hand_joint_detector, feet_joint_detector):
        super().__init__(config, consider_flip = True)

        self.hand_joint_detector = hand_joint_detector
        self.feet_joint_detector = feet_joint_detector

    def create_dream_datasets(self, img_dir, file_names = None):
        full_dataframe = self._create_detection_dataframe(img_dir, file_names)

        hands_mask = ['H' in image_name for image_name in full_dataframe['image_name']]

        hand_dataframe = full_dataframe.iloc[hands_mask]
        feet_dataframe = full_dataframe.iloc[np.logical_not(hands_mask)]

        data_hands = self._detect_joints_in_image_data(hand_dataframe, img_dir, self.hand_joint_detector, hand_coord_mapping)
        data_feet = self._detect_joints_in_image_data(feet_dataframe, img_dir, self.feet_joint_detector, foot_coord_mapping)

        return data_hands, data_feet

    def _add_dream_outputs(self, output_mapping):
        def _create_dream_output_lambda(image_name):
            part = image_name.split('-')[1]

            def __add_dream_outputs(key, output_dict, row):
                joint_mapping = output_mapping[key]

                mapped_keys = [key_val.format(part = part) for key_val in joint_mapping[0]]
                for idx, mapped_key in enumerate(mapped_keys):
                    output_dict[f'narrowing_{idx}'] = row[mapped_key]

                mapped_keys = [key_val.format(part = part) for key_val in joint_mapping[1]]
                for idx, mapped_key in enumerate(mapped_keys):
                    output_dict[f'erosion_{idx}'] = row[mapped_key]

                return output_dict

            return lambda key, output_dict, row: __add_dream_outputs(key, output_dict, row)

        return lambda image_name: _create_dream_output_lambda(image_name)

    def _get_dataframes(self, training_csv):
        info = pd.read_csv(training_csv)
        features = info.columns
        parts = ["LH","RH","LF","RF"]
        dataframes = {}
        for part in parts:
            flip = 'N'
            if(part.startswith('R')):
                flip = 'Y'
            
            dataframes[part] = info.loc[:,["Patient_ID"]+[s for s in features if part in s]].copy()
            dataframes[part]["Patient_ID"] = dataframes[part]["Patient_ID"].astype(str) + f"-{part}"
            dataframes[part]["flip"] = flip
            dataframes[part]["file_type"] = 'jpg'
            
        data_hands = pd.concat((dataframes["RH"],dataframes["LH"]), sort = False)
        data_feet = pd.concat((dataframes["RF"],dataframes["LF"]), sort = False)
            
        return data_hands, data_feet

class rsna_joint_detector(joint_detector):
    def __init__(self, config, hand_joint_detector):
        super().__init__(config, '../rsna_boneAge/checked_rsna_training')

        self.training_csv = '../rsna_boneAge/boneage-training-dataset.csv'
        self.hand_joint_detector = hand_joint_detector

    def create_rnsa_dataset(self):
        key_columns = ['id', 'file_type', 'flip']

        dataframe = self._load_dataframe()
        dataframe = self._detect_joints_in_image_data(dataframe, key_columns, self.hand_joint_detector)

        return self._create_output_dataframe(dataframe, hand_coord_mapping, self._add_rnsa_outputs())

    def _add_rnsa_outputs(self):
        def __add_rnsa_outputs(key, output_dict, row):
            output_dict['boneage'] = row['boneage']
            output_dict['sex'] = row['male']

            return output_dict

        def _create_rnsa_output_lambda(image_name):
            return lambda key, output_dict, row: __add_rnsa_outputs(key, output_dict, row)

        return lambda image_name: _create_rnsa_output_lambda(image_name)

    def _load_dataframe(self):
        #Go through each image in the checked directory
        checked_rnsa_images = os.listdir(self.image_directory)

        rnsa_dicts = []
        for checked_rnsa_image in checked_rnsa_images:
            file_name_parts = checked_rnsa_image.split('.')

            file_name = file_name_parts[0]
            file_type = file_name_parts[1]

            rnsa_dict = {
                'id': file_name,
                'file_type': file_type, 
                'flip': 'N'
            }

            rnsa_dicts.append(rnsa_dict)

        # Create dataframe from read images
        rnsa_checked_images_dataframe = pd.DataFrame(rnsa_dicts, dtype = np.str, index = np.arange(len(rnsa_dicts)))

        # Read full outcomes from *.csv, and encode gender as 0/1 (0 == Female)
        outcomes_dataframe = pd.read_csv(self.training_csv)
        outcomes_dataframe = outcomes_dataframe.astype({'id': 'str', 'male': 'int32'})

        # Return merged dataframe that only contains entries with checked images and outcomes
        return rnsa_checked_images_dataframe.merge(outcomes_dataframe, how = 'inner', on = 'id')
