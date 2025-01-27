import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataset.base_dataset import joint_dataset

hand_joint_keys = ['mcp', 'pip_2', 'pip_3', 'pip_4', 'pip_5', 'mcp_1', 'mcp_2', 'mcp_3', 'mcp_4', 'mcp_5']

hand_wrist_keys = ['w1', 'w2', 'w3']

class rsna_joint_dataset(joint_dataset):
    def __init__(self, config, imagenet = False, pad_resize = False, joint_extractor = None):
        super().__init__(config, 'rsna_joints', imagenet = imagenet, pad_resize = pad_resize, joint_extractor = joint_extractor)

        self.image_dir = config.rsna_img_dir
        self.outcomes_source = config.rsna_labels
        self.cache = self.cache + 'rsna/'

    def create_rsna_joints_dataset(self, joints_source = './data/predictions/rsna_joint_data_v3.csv', val_split = False, include_wrist_joints = True):
        joint_keys = hand_joint_keys
        if(include_wrist_joints):
            joint_keys.extend(hand_wrist_keys)

        outcomes_df = self._create_intermediate_outcomes_df(self.image_dir, joint_keys) 

        joints_df = self._create_intermediate_joints_df(joints_source, joint_keys)
        joints_df = joints_df.astype({'image_name': 'str'})
        
        outcome_joint_df = outcomes_df.merge(joints_df, left_on = ['id', 'key'], right_on = ['image_name', 'key'], how = "inner")

        return self._create_rsna_datasets(outcome_joint_df, val_split)

    def _create_intermediate_outcomes_df(self, image_dir, keys):
        rsna_images = os.listdir(image_dir)

        rsna_dicts = []
        for rsna_image in rsna_images:
            file_info = rsna_image.split('.')

            file_name = file_info[0]

            for joint_type in keys:
                rsna_dict = {
                    'id': file_name,
                    'key': joint_type
                }

                rsna_dicts.append(rsna_dict)

        rsna_img_df = pd.DataFrame(rsna_dicts, dtype = np.str, index = np.arange(len(rsna_dicts)))

        outcomes_df = pd.read_csv(self.outcomes_source)
        outcomes_df = outcomes_df.astype({'id': 'str', 'male': 'int32'})

        return rsna_img_df.merge(outcomes_df, on = 'id')

    def _create_rsna_datasets(self, outcomes_df, val_split = False, wrist = False):
        outcomes_df = outcomes_df.sample(frac = 1).reset_index(drop = True)
        
        file_info = outcomes_df[['image_name', 'file_type', 'flip', 'key']].astype(np.str).to_numpy()
        if wrist:
            coords = outcomes_df[['w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']].to_numpy()
        else:
            coords = outcomes_df[['coord_x', 'coord_y']].to_numpy()

        outcomes = outcomes_df[['boneage', 'male', 'key']]
        outcomes = pd.get_dummies(outcomes, columns = ['key'], dtype = np.float32)
        outcomes = outcomes.to_numpy(np.float64)

        if(val_split):
            file_info, file_test, coords, coords_test, outcomes, outcomes_test = train_test_split(file_info, coords, outcomes, test_size = 0.1)

            rsna_dataset = self._create_non_split_joint_dataset(file_info, coords, outcomes, cache = self.cache, wrist = wrist, buffer_size = 4000)
            rsna_val_dataset = self._create_non_split_joint_dataset(file_test, coords_test, outcomes_test, augment = False, wrist = wrist)
            
            return rsna_dataset, rsna_val_dataset
        else:
            return self._create_non_split_joint_dataset(file_info, coords, outcomes, cache = self.cache, wrist = wrist, buffer_size = 4000)

class rsna_wrist_dataset(rsna_joint_dataset):
    def __init__(self, config, imagenet = False, pad_resize = False):
        joint_dataset.__init__(self, config, 'rsna_wrists', imagenet = imagenet, pad_resize = pad_resize)

        self.image_dir = config.rsna_img_dir
        self.outcomes_source = config.rsna_labels

    def create_rsna_wrist_dataset(self, joints_source = './data/predictions/rsna_joint_data_v3.csv', val_split = False):
        outcomes_df = self._create_intermediate_outcomes_wrists_df(self.image_dir, hand_wrist_keys) 
        joints_df = self._create_intermediate_wrists_df(joints_source, hand_wrist_keys)
        joints_df = joints_df.astype({'image_name': 'str'})
        
        outcome_joint_df = outcomes_df.merge(joints_df, left_on = ['id', 'key'], right_on = ['image_name', 'key'], how = "inner")

        return self._create_rsna_datasets(outcome_joint_df, val_split, wrist = True)

    def _create_intermediate_outcomes_wrists_df(self, image_dir, keys):
        rsna_images = os.listdir(image_dir)

        rsna_dicts = []
        for rsna_image in rsna_images:
            file_info = rsna_image.split('.')
            file_name = file_info[0]
            rsna_dict = {
                'id': file_name,
                'key': "wrist"
            }

            rsna_dicts.append(rsna_dict)

        rsna_img_df = pd.DataFrame(rsna_dicts, dtype = np.str, index = np.arange(len(rsna_dicts)))

        outcomes_df = pd.read_csv(self.outcomes_source)
        outcomes_df = outcomes_df.astype({'id': 'str', 'male': 'int32'})

        return rsna_img_df.merge(outcomes_df, on = 'id')
