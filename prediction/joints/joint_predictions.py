import os

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.ops.image_ops as img_ops
import dataset.ops.landmark_ops as lm_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Map each joint, to its indicies in the predicted landmarks vector, and its outcome scores
hand_mapping = {
    'mcp': [[0, 2], [], ['{part}_mcp_E__ip'] ],
    'pip_2': [[2, 4], ['{part}_pip_J__2'], ['{part}_pip_E__2']],
    'pip_3': [[4, 6], ['{part}_pip_J__3'], ['{part}_pip_E__3']],
    'pip_4': [[6, 8], ['{part}_pip_J__4'], ['{part}_pip_E__4']],
    'pip_5': [[8, 10], ['{part}_pip_J__5'], ['{part}_pip_E__5']],
    'mcp_1': [[10, 12], ['{part}_mcp_J__1'], ['{part}_mcp_E__1']],
    'mcp_2': [[12, 14], ['{part}_mcp_J__2'], ['{part}_mcp_E__2']],
    'mcp_3': [[14, 16], ['{part}_mcp_J__3'], ['{part}_mcp_E__3']],
    'mcp_4': [[16, 18], ['{part}_mcp_J__4'], ['{part}_mcp_E__4']],
    'mcp_5': [[18, 20], ['{part}_mcp_J__5'], ['{part}_mcp_E__5']],
    'w1': [[20, 22], ['{part}_wrist_J__radcar', '{part}_wrist_J__mna'], ['{part}_wrist_E__mc1', '{part}_wrist_E__mul']],
    'w2': [[22, 24], ['{part}_wrist_J__cmc3', '{part}_wrist_J__capnlun'], ['{part}_wrist_E__radius', '{part}_wrist_E__nav']],
    'w3': [[24, 26], ['{part}_wrist_J__cmc4', '{part}_wrist_J__cmc5'], ['{part}_wrist_E__ulna', '{part}_wrist_E__lunate']]
}

feet_mapping = {
    'mtp': [[0, 2], ['{part}_mtp_J__ip'], ['{part}_mtp_E__ip']], 
    'mtp_1': [[2, 4], ['{part}_mtp_J__1'], ['{part}_mtp_E__1']], 
    'mtp_2': [[4, 6], ['{part}_mtp_J__2'], ['{part}_mtp_E__2']],
    'mtp_3': [[6, 8], ['{part}_mtp_J__3'], ['{part}_mtp_E__3']],
    'mtp_4': [[8, 10], ['{part}_mtp_J__4'], ['{part}_mtp_E__4']],
    'mtp_5': [[10, 12], ['{part}_mtp_J__5'], ['{part}_mtp_E__5']]
}

class dream_joint_detector():
    def __init__(self, config, hand_joint_detector, feet_joint_detector):
        self.image_directory = config.train_location
        self.img_width = config.landmarks_img_width
        self.img_height = config.landmarks_img_height

        self.hand_joint_detector = hand_joint_detector
        self.feet_joint_detector = feet_joint_detector

    def create_dream_datasets(self, source_file = 'training.csv'):
        data_hands, data_feet = self._get_dataframes(os.path.join(self.image_directory, source_file))

        data_hands = self._add_joint_predictions(data_hands, self.hand_joint_detector)
        data_feet = self._add_joint_predictions(data_feet, self.feet_joint_detector)

        return self._create_output_dataframe(data_hands, hand_mapping), self._create_output_dataframe(data_feet, feet_mapping)

    def _add_joint_predictions(self, data_frame, joint_detector):
        joint_prediction_dataset = self._create_joint_prediction_dataset(data_frame)

        joint_predictions_list = []
        for patient_id, landmark_image, original_image_shape in joint_prediction_dataset:
            joint_predictions = joint_detector.predict(landmark_image)

            joint_prediction = {
                'Patient_ID': patient_id.numpy().decode('UTF-8'),
                'joint_locations': joint_predictions,
                'original_shape': original_image_shape
            }

            joint_predictions_list.append(joint_prediction)

        joint_dataframe = pd.DataFrame(joint_predictions_list, index = np.arange(len(joint_predictions_list)))

        return data_frame.merge(joint_dataframe, on = 'Patient_ID')

    def _create_output_dataframe(self, data_frame, joints_mapping):
        joint_dicts = []
        
        for _, row in data_frame.iterrows():
            patient_id = row['Patient_ID']
            part = patient_id.split('-')[1]
            
            joint_locations = row['joint_locations'][0]

            joint_locations = lm_ops.upscale_detected_landmarks(joint_locations, (self.img_height, self.img_height), row['original_shape'])

            for key in joints_mapping:
                joint_mapping = joints_mapping[key]

                coord_idxs = joint_mapping[0]
                joint_coords = joint_locations[coord_idxs[0]:coord_idxs[1]].numpy()

                joint_dict = {
                    'image_name': row['Patient_ID'],
                    'key': key,
                    'coord_x': joint_coords[0],
                    'coord_y': joint_coords[1]
                }
                
                mapped_keys = [key_val.format(part = part) for key_val in joint_mapping[1]]
                for idx, mapped_key in enumerate(mapped_keys):
                    joint_dict[f'narrowing_{idx}'] = row[mapped_key]

                mapped_keys = [key_val.format(part = part) for key_val in joint_mapping[2]]
                for idx, mapped_key in enumerate(mapped_keys):
                    joint_dict[f'erosion_{idx}'] = row[mapped_key]

                joint_dicts.append(joint_dict)
        
        return pd.DataFrame(joint_dicts, np.arange(len(joint_dicts)))

    def _create_joint_prediction_dataset(self, data_frame):
        x = data_frame[['Patient_ID', 'flip']].to_numpy()

        dataset = tf.data.Dataset.from_tensor_slices((x))
        dataset = self.get_landmark_detection_image(dataset)
        dataset = dataset.prefetch(buffer_size = AUTOTUNE)

        return dataset 

    def get_landmark_detection_image(self, dataset):
        def __load_joints(file):
            file_name = file[0]
            flip = file[1]
        
            flip_img = flip == 'Y'

            image, _ = img_ops.load_image(file_name, [], False, self.image_directory, flip_img)
            landmark_detection_image, _ = img_ops.resize_image(image, [], False, self.img_width, self.img_height)

            return file_name, tf.expand_dims(landmark_detection_image, 0), tf.shape(image)

        return dataset.map(__load_joints, num_parallel_calls = AUTOTUNE)

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
            # dataframes[part]["total_fig_score"] = dataframes[part].loc[:,[s for s in features if part in s]].sum(axis=1)
            dataframes[part]["Patient_ID"] = dataframes[part]["Patient_ID"].astype(str) + f"-{part}"
            dataframes[part]["flip"] = flip
            
        data_hands = pd.concat((dataframes["RH"],dataframes["LH"]), sort = False)
        data_feet = pd.concat((dataframes["RF"],dataframes["LF"]), sort = False)
            
        return data_hands, data_feet