import numpy as np
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import dataset.ops.joint_ops as joint_ops
import dataset.ops.dataset_ops as ds_ops

from dataset.base_dataset import base_dataset

dream_hand_parts = ['LH', 'RH']
dream_foot_parts = ['LF', 'RF']

foot_outcome_mapping = {
    'mtp': [['{part}_mtp_J__ip'], ['{part}_mtp_E__ip']], 
    'mtp_1': [['{part}_mtp_J__1'], ['{part}_mtp_E__1']], 
    'mtp_2': [['{part}_mtp_J__2'], ['{part}_mtp_E__2']],
    'mtp_3': [['{part}_mtp_J__3'], ['{part}_mtp_E__3']],
    'mtp_4': [['{part}_mtp_J__4'], ['{part}_mtp_E__4']],
    'mtp_5': [['{part}_mtp_J__5'], ['{part}_mtp_E__5']]
}

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
}

wrist_outcome_mapping = {
    "wrist": [['{part}_wrist_J__radcar', '{part}_wrist_J__mna', '{part}_wrist_J__cmc3', '{part}_wrist_J__capnlun', '{part}_wrist_J__cmc4', '{part}_wrist_J__cmc5'],
             ['{part}_wrist_E__mc1', '{part}_wrist_E__mul', '{part}_wrist_E__radius', '{part}_wrist_E__nav', '{part}_wrist_E__ulna', '{part}_wrist_E__lunate']]
}

hand_joint_keys = ['mcp', 'pip_2', 'pip_3', 'pip_4', 'pip_5', 'mcp_1', 'mcp_2', 'mcp_3', 'mcp_4', 'mcp_5']

hand_wrist_keys = ['w1', 'w2', 'w3']

class joint_dataset(base_dataset):
    def __init__(self, config, cache_postfix):
        super().__init__(config)

        self.cache = config.cache_loc + cache_postfix
        self.image_dir = config.train_location + '/fixed'
        self.joint_height = config.joint_img_height
        self.joint_width = config.joint_img_width

    def _create_dataset(self, file_info, joint_coords, outcomes, wrist = False, augment = True, cache = True):
        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords, outcomes))
        if wrist:
            dataset = joint_ops.load_wrists(dataset, self.image_dir)
        else:
            dataset = joint_ops.load_joints(dataset, self.image_dir)
        
        return self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, cache = cache, pad_resize = False, augment = augment)

    def _create_intermediate_joints_df(self, joints_source, joint_keys):
        joints_df = pd.read_csv(joints_source)

        mapped_joints = []
        for _, row in joints_df.iterrows():
            image_name = row['image_name']
  
            for key in joint_keys:
                mapped_joint = {
                    'image_name': image_name,
                    'key': key,
                    'flip': row['flip'],
                    'file_type': row['file_type'],
                    'coord_x': row[key + '_x'],
                    'coord_y': row[key + '_y']
                }

                mapped_joints.append(mapped_joint)

        return pd.DataFrame(mapped_joints, index = np.arange(len(mapped_joints)))
        
    def _create_intermediate_wrists_df(self, joints_source, joint_keys):
        joints_df = pd.read_csv(joints_source)

        mapped_joints = []
        for _, row in joints_df.iterrows():
            image_name = row['image_name']
            mapped_joint = {
                'image_name': image_name,
                'key': "wrist",
                'flip': row['flip'],
                'file_type': row['file_type'],
            }

            for key in joint_keys:
                mapped_joint[key + "_x"] = row[key + '_x']
                mapped_joint[key + "_y"] = row[key + '_y']
                
            mapped_joints.append(mapped_joint)

        return pd.DataFrame(mapped_joints, index = np.arange(len(mapped_joints)))

class dream_dataset(joint_dataset):
    def __init__(self, config, cache_postfix):
        super().__init__(config, cache_postfix)

    def _create_dream_datasets(self, outcomes_source, joints_source, val_joints_source, outcome_mapping, parts, outcome_columns, wrist = False):
        outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, joints_source, outcome_mapping, parts, wrist=wrist)
        outcome_joint_df = outcome_joint_df.dropna(subset=outcome_columns)
        if(val_joints_source):
            outcome_joint_val_df = self._create_outcome_joint_dataframe(outcomes_source, val_joints_source, outcome_mapping, parts, wrist=wrist)
            outcome_joint_val_df = outcome_joint_val_df.dropna(subset=outcome_columns)

        dataset = self._create_dream_dataset(outcome_joint_df, outcome_columns, cache = self.cache, wrist=wrist)
        if val_joints_source:
            val_dataset = self._create_dream_dataset(outcome_joint_val_df, outcome_columns, is_train = False, augment = False, wrist=wrist)

            return dataset, val_dataset
        else:
            return dataset

    def _create_outcome_joint_dataframe(self, outcomes_source, joints_source, outcome_mapping, parts, wrist = False):
        outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, outcome_mapping, parts)
        if wrist:
            joints_df = self._create_intermediate_wrists_df(joints_source, hand_wrist_keys)
        else:
            joints_df = self._create_intermediate_joints_df(joints_source, outcome_mapping.keys())

        return outcomes_df.merge(joints_df, on = ['image_name', 'key'])

    def _create_intermediate_outcomes_df(self, outcomes_source, outcome_mapping, parts):
        outcomes_df = pd.read_csv(outcomes_source)

        outcome_joints = []
        for _, row in outcomes_df.iterrows():
            patient_id = row['Patient_ID']

            for part in parts:
                image_name = patient_id + '-' + part

                for idx, key in enumerate(outcome_mapping.keys()):
                    joint_mapping = outcome_mapping[key]

                    outcome_joint = {
                        'image_name': image_name,
                        'key': key
                    }
                    
                    mapped_narrowing_keys = [key_val.format(part = part) for key_val in joint_mapping[0]]
                    for idx, mapped_key in enumerate(mapped_narrowing_keys):
                        outcome_joint[f'narrowing_{idx}'] = row[mapped_key]

                    mapped_erosion_keys = [key_val.format(part = part) for key_val in joint_mapping[1]]
                    for idx, mapped_key in enumerate(mapped_erosion_keys):
                        outcome_joint[f'erosion_{idx}'] = row[mapped_key]

                    outcome_joints.append(outcome_joint)

        return pd.DataFrame(outcome_joints, index = np.arange(len(outcome_joints)))    

    def _create_dream_dataset(self, outcome_joint_df, outcome_columns, is_train = True, augment = True, cache = True, wrist = False):
        file_info = outcome_joint_df[['image_name', 'file_type', 'flip', 'key']].values

        outcomes = outcome_joint_df[outcome_columns]
        if is_train:
            self._init_model_outcomes_bias(outcomes)
            
        outcomes = pd.get_dummies(outcomes, columns = outcome_columns)
        
        if wrist:
            coords = outcome_joint_df[['w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']].values
        else:
            coords = outcome_joint_df[['coord_x', 'coord_y']].values

        return self._create_dataset(file_info, coords, outcomes.to_numpy(dtype = np.float64), augment = augment, cache = cache, wrist = wrist)

    def _init_model_outcomes_bias(self, outcomes):
        N, D = outcomes.shape

        outcomes_class_weights = []
        outcomes_class_bias = []

        for _ in range(D):
            classes, counts = np.unique(outcomes.to_numpy(), return_counts = True)

            weights = (1 / counts) * (N) / 2.0

            class_weights = {}
            for idx, c in enumerate(classes.astype(np.int64)):
                class_weights[c] = weights[idx]

            outcomes_class_weights.append(class_weights)
            outcomes_class_bias.append(np.log(counts / np.sum(counts)))

        self.class_weights = outcomes_class_weights
        self.class_bias = outcomes_class_bias

class feet_joint_dataset(dream_dataset):
    def __init__(self, config):
        super().__init__(config, 'feet_joints')

        self.image_dir = config.train_fixed_location

    def create_feet_joints_dataset(self, outcomes_source, joints_source = './data/predictions/feet_joint_data.csv', val_joints_source = None, erosion_flag = False):
        outcome_column = 'narrowing_0'
        if(erosion_flag):
            outcome_column = 'erosion_0'

        return self._create_dream_datasets(outcomes_source, joints_source, val_joints_source, foot_outcome_mapping, dream_foot_parts, [outcome_column])


class hands_joints_dataset(dream_dataset):
    def __init__(self, config):
        super().__init__(config, 'hands_joints')

        self.image_dir = config.train_fixed_location

    def create_hands_joints_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data.csv', val_joints_source = None, erosion_flag = False):
        outcome_column = 'narrowing_0'
        if(erosion_flag):
            outcome_column = 'erosion_0'

        return self._create_dream_datasets(outcomes_source, joints_source, val_joints_source, hand_outcome_mapping, dream_hand_parts, [outcome_column])


class hands_wrists_dataset(dream_dataset):
    def __init__(self, config):
        super().__init__(config, 'wrists_joints')

        self.image_dir = config.train_fixed_location

    def create_wrists_joints_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data.csv', val_joints_source = None, erosion_flag = False):
        outcome_columns = ['narrowing_0', 'narrowing_1', 'narrowing_2', 'narrowing_3', 'narrowing_4', 'narrowing_5']
        if(erosion_flag):
            outcome_columns = ['erosion_0', 'erosion_1', 'erosion_2', 'erosion_3', 'erosion_4', 'erosion_5']

        return self._create_dream_datasets(outcomes_source, joints_source, val_joints_source, wrist_outcome_mapping, dream_hand_parts, outcome_columns, wrist=True)
