import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataset.joint_dataset import joint_dataset

hand_joint_keys = ['mcp', 'pip_2', 'pip_3', 'pip_4', 'pip_5', 'mcp_1', 'mcp_2', 'mcp_3', 'mcp_4', 'mcp_5', 'w1', 'w2', 'w3']

class rsna_joint_dataset(joint_dataset):
    def __init__(self, config):
        super().__init__(config, 'rsna_joints')

        self.image_dir = config.rsna_img_dir
        self.outcomes_source = config.rsna_labels

    def create_rsna_joints_dataset(self, joints_source = './data/predictions/rsna_joint_data.csv', val_split = False):
        outcomes_df = self._create_intermediate_outcomes_df(self.image_dir)
        joints_df = self._create_intermediate_joints_df(joints_source, hand_joint_keys)
        joints_df = joints_df.astype({'image_name': 'str'})
        
        outcome_joint_df = outcomes_df.merge(joints_df, left_on = ['id', 'key'], right_on = ['image_name', 'key'])

        return self._create_rsna_datasets(outcome_joint_df, val_split)

    def _create_intermediate_outcomes_df(self, image_dir):
        rsna_images = os.listdir(image_dir)

        rsna_dicts = []
        for rsna_image in rsna_images:
            file_info = rsna_image.split('.')

            file_name = file_info[0]

            for joint_type in hand_joint_keys:
                rsna_dict = {
                    'id': file_name,
                    'key': joint_type
                }

                rsna_dicts.append(rsna_dict)

        rsna_img_df = pd.DataFrame(rsna_dicts, dtype = np.str, index = np.arange(len(rsna_dicts)))

        outcomes_df = pd.read_csv(self.outcomes_source)
        outcomes_df = outcomes_df.astype({'id': 'str', 'male': 'int32'})

        return rsna_img_df.merge(outcomes_df, on = 'id')

    def _create_rsna_datasets(self, outcomes_df, val_split = False):
        file_info = outcomes_df[['image_name', 'file_type', 'flip', 'key']].astype(np.str).to_numpy()
        coords = outcomes_df[['coord_x', 'coord_y']].to_numpy()

        outcomes = outcomes_df[['boneage', 'male', 'key']]
        outcomes = pd.get_dummies(outcomes, columns = ['key'], dtype = np.float32)
        outcomes = outcomes.to_numpy(np.float64)

        if(val_split):
            file_info, file_test, coords, coords_test, outcomes, outcomes_test = train_test_split(file_info, coords, outcomes, test_size = 0.2)

            rsna_dataset = self._create_dataset(file_info, coords, outcomes, cache = self.cache)
            rsna_val_dataset = self._create_dataset(file_test, coords_test, outcomes_test, augment = False)
            
            return rsna_dataset, rsna_val_dataset
        else:
            return self._create_dataset(file_info, coords, outcomes, cache = self.cache)