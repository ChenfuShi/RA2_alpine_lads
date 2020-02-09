import os

import numpy as np
import pandas as pd

from prediction.joints import joint_detector

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

class dream_joint_detector(joint_detector):
    def __init__(self, config, hand_joint_detector, feet_joint_detector):
        super().__init__(config.train_location, config.landmarks_img_width, config.landmarks_img_height)

        self.hand_joint_detector = hand_joint_detector
        self.feet_joint_detector = feet_joint_detector

    def create_dream_dataframes(self, source_file = 'training.csv'):
        training_data = pd.read_csv(os.path.join(self.image_directory, source_file))

        hands = ['LH', 'RH']
        feet = ['LF', 'RF']
        
        hand_dirs = []
        feet_dirs = []

        for _, row in training_data.iterrows():
            for part in hands:
                hand_joints = self._locate_joints_and_outputs(row, part, self.hand_joint_detector, hand_mapping)
                
                hand_dirs.extend(hand_joints)

            for part in feet:
                foot_joints = self._locate_joints_and_outputs(row, part, self.feet_joint_detector, feet_mapping)

                feet_dirs.extend(foot_joints)

        hand_dataframe = pd.DataFrame(hand_dirs, index = np.arange(len(hand_dirs)))
        feet_dataframe = pd.DataFrame(feet_dirs, index = np.arange(len(feet_dirs)))
        
        return hand_dataframe, feet_dataframe

    # Given a row, and one part, this loads the image for this part, finds the joints, and collects the outputs
    def _locate_joints_and_outputs(self, row, part, detector, joints_mapping):
        patient_id = row['Patient_ID']
        image_name = patient_id + f"-{part}"
        
        flip_img = 'R' in part

        flip_flag = 'N'
        if(flip_img):
            flip_flag = 'Y'

        predicted_landmarks = self._get_joint_landmark_coords(image_name, flip_img, detector)

        joint_dicts = []
        for key in joints_mapping:
            joint_mapping = joints_mapping[key]

            coord_idxs = joint_mapping[0]
            joint_coords = predicted_landmarks[coord_idxs[0]:coord_idxs[1]].numpy()

            joint_dict = {
                'image_name': image_name,
                'key': key,
                'flip': flip_flag,
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

        return joint_dicts