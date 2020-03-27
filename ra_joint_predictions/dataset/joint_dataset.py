import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import dataset.ops.joint_ops as joint_ops
import dataset.ops.dataset_ops as ds_ops
import model.joint_damage_model as joint_damage_model

from dataset.base_dataset import base_dataset
from utils.class_weight_utils import calc_adapted_class_weights

dream_hand_parts = ['LH', 'RH']
dream_foot_parts = ['LF', 'RF']

hands_narrowing_params = {
    'no_classes': 5,
    'outcomes': ['narrowing_0'],
    'parts': dream_hand_parts
}

wrists_narrowing_params = {
    'no_classes': 5,
    'outcomes': ['narrowing_0', 'narrowing_1', 'narrowing_2', 'narrowing_3', 'narrowing_4', 'narrowing_5'],
    'parts': dream_hand_parts
}

feet_narrowing_params = {
    'no_classes': 5,
    'outcomes': ['narrowing_0'],
    'parts': dream_foot_parts
}

hands_erosion_params = {
    'no_classes': 6,
    'outcomes': ['erosion_0'],
    'parts': dream_hand_parts
}

wrists_erosion_params = {
    'no_classes': 6,
    'outcomes': ['erosion_0', 'erosion_1', 'erosion_2', 'erosion_3', 'erosion_4', 'erosion_5'],
    'parts': dream_hand_parts
}

feet_erosion_params = {
    'no_classes': 11,
    'outcomes': ['erosion_0'],
    'parts': dream_foot_parts
}

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

AUTOTUNE = tf.data.experimental.AUTOTUNE

class joint_dataset(base_dataset):
    def __init__(self, config, cache_postfix = '', imagenet = False, pad_resize = False, joint_extractor = None):
        super().__init__(config)
        self.imagenet = imagenet
        self.cache = config.cache_loc + cache_postfix
        self.joint_height = config.joint_img_height
        self.joint_width = config.joint_img_width
        self.pad_resize = pad_resize
        self.joint_extractor = joint_extractor

    def _create_joint_dataset(self, file_info, joint_coords, outcomes, wrist = False):
        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords, outcomes))
        if wrist:
            dataset = joint_ops.load_wrists(dataset, self.image_dir, imagenet = self.imagenet)
        else:
            dataset = joint_ops.load_joints(dataset, self.image_dir, imagenet = self.imagenet, joint_extractor = self.joint_extractor)
        
        return dataset

    def _create_non_split_joint_dataset(self, file_info, coords, outcomes, cache = True, wrist = False, augment = True, buffer_size = 200):
        dataset = self._create_joint_dataset(file_info, coords, outcomes, wrist)

        dataset = self._cache_shuffle_repeat_dataset(dataset, cache = cache, buffer_size = buffer_size)
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize, augment = augment)
        
        return dataset

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
    def __init__(self, config, cache_postfix = '', model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, cache_postfix, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location
        self.batch_size = config.batch_size
        self.model_type = model_type
        self.cache = self.cache + '_' + model_type

    def _create_dream_datasets(self, outcomes_source, joints_source, outcome_mapping, parts, outcome_columns, no_classes, wrist = False):
        outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, joints_source, outcome_mapping, parts, wrist = wrist)
        outcome_joint_df = outcome_joint_df.dropna(subset = outcome_columns)

        dataset = self._create_dream_dataset(outcome_joint_df, outcome_columns, no_classes, cache = self.cache, wrist = wrist)
        
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

    def _create_dream_dataset(self, outcome_joint_df, outcome_columns, no_classes, augment = True, cache = True, wrist = False, is_train = True):
        outcome_joint_df = outcome_joint_df.sample(frac = 1).reset_index(drop = True)

        file_info = outcome_joint_df[['image_name', 'file_type', 'flip', 'key']].values

        outcomes = outcome_joint_df[outcome_columns]
        if is_train:
            self._init_model_outcomes_bias(outcomes, no_classes)

        tf_dummy_outcomes = None
        tf_outcomes = None

        if self.model_type == joint_damage_model.MODEL_TYPE_CLASSIFICATION:
            tf_dummy_outcomes = self._dummy_encode_outcomes(outcomes, no_classes)
        elif self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
            tf_outcomes = outcomes.to_numpy()
        elif self.model_type == joint_damage_model.MODEL_TYPE_COMBINED:
            tf_dummy_outcomes = self._dummy_encode_outcomes(outcomes, no_classes)
            tf_outcomes = outcomes.to_numpy()
        
        if wrist:
            coords = outcome_joint_df[['w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']].values
        else:
            coords = outcome_joint_df[['coord_x', 'coord_y']].values

        if is_train:
            maj_idx = self._find_maj_indices(outcomes)

            return self._create_interleaved_joint_datasets(file_info, coords, maj_idx, outcomes = tf_outcomes, dummy_outcomes = tf_dummy_outcomes, wrist = wrist)
        else:
            return self._create_non_split_joint_dataset(file_info, coords, tf_outcomes, wrist = wrist, augment = False)

    def _find_maj_indices(self, outcomes):
        return outcomes == 0

    def _create_interleaved_joint_datasets(self, file_info, joint_coords, maj_idx, outcomes = None, dummy_outcomes = None, wrist = False):
        min_idx = np.logical_not(maj_idx)

        # Tranform boolean mask into indices
        maj_idx = np.where(maj_idx)[0]
        min_idx = np.where(min_idx)[0]

        if self.model_type == 'C':
            maj_outcomes = dummy_outcomes[maj_idx, :]
            min_outcomes = dummy_outcomes[min_idx, :]
        elif self.model_type == 'R':
            maj_outcomes = outcomes[maj_idx]
            min_outcomes = outcomes[min_idx]
        elif self.model_type == 'RC':
            maj_outcomes = (outcomes[maj_idx], dummy_outcomes[maj_idx, :])
            min_outcomes = (outcomes[min_idx], dummy_outcomes[min_idx, :])

        # Create 2 datasets, one with the majority class, one with the other classes
        maj_ds = self._create_joint_dataset(file_info[maj_idx, :], joint_coords[maj_idx], maj_outcomes, wrist = wrist)
        min_ds = self._create_joint_dataset(file_info[min_idx, :], joint_coords[min_idx], min_outcomes, wrist = wrist)

        # Cache the partial datasets, shuffle the datasets with buffersize that ensures minority samples are all shuffled
        maj_ds = self._cache_shuffle_repeat_dataset(maj_ds, self.cache + '_maj', buffer_size = min_idx.shape[0])
        min_ds = self._cache_shuffle_repeat_dataset(min_ds, self.cache + '_min', buffer_size = min_idx.shape[0])

        # Interleave datasets 50/50 - for each majority sample (class 0), it adds one none majority sample (not class 0)
        dataset = tf.data.experimental.sample_from_datasets((maj_ds, min_ds), [0.5, 0.5])

        # Prepare for training
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize)

        return dataset

    def _init_model_outcomes_bias(self, outcomes, no_classes):
        self.class_weights = calc_adapted_class_weights(outcomes, no_classes)

    def _dummy_encode_outcomes(self, outcomes, no_classes):
        D = outcomes.shape[1]

        one_hot_encoder = OneHotEncoder(sparse = False, categories = [np.arange(no_classes)] * D)
        column_transformer = ColumnTransformer([('one_hot_encoder', one_hot_encoder, np.arange(D))], remainder = 'passthrough')

        return column_transformer.fit_transform(outcomes.to_numpy()).astype(dtype = np.float64)

class feet_joint_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, 'feet_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location

    def create_feet_joints_dataset(self, outcomes_source, joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = False):
        outcome_column = 'narrowing_0'
        no_classes = 5
        self.cache = self.cache + '_narrowing'
        if(erosion_flag):
            outcome_column = 'erosion_0'
            no_classes = 11
            self.cache = self.cache + '_erosion'

        return self._create_dream_datasets(outcomes_source, joints_source, foot_outcome_mapping, dream_foot_parts, [outcome_column], no_classes)

class hands_joints_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, 'hands_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location

    def create_hands_joints_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_v2.csv', erosion_flag = False):
        outcome_column = 'narrowing_0'
        no_classes = 5
        self.cache = self.cache + '_narrow'
        if(erosion_flag):
            outcome_column = 'erosion_0'
            no_classes = 6
            self.cache = self.cache + '_erosion'

        return self._create_dream_datasets(outcomes_source, joints_source, hand_outcome_mapping, dream_hand_parts, [outcome_column], no_classes)

class hands_wrists_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, 'wrists_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location

    def create_wrists_joints_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_v2.csv', erosion_flag = False):
        outcome_columns = ['narrowing_0', 'narrowing_1', 'narrowing_2', 'narrowing_3', 'narrowing_4', 'narrowing_5']
        no_classes = 5
        self.cache = self.cache + '_narrow'
        if(erosion_flag):
            outcome_columns = ['erosion_0', 'erosion_1', 'erosion_2', 'erosion_3', 'erosion_4', 'erosion_5']
            no_classes = 6
            self.cache = self.cache + '_erosion'
            
        dataset = self._create_dream_datasets(outcomes_source, joints_source, wrist_outcome_mapping, dream_hand_parts, outcome_columns, no_classes, wrist = True)

        return self._split_outcomes(dataset, no_classes)

    # Overwrite method for wrist dataset to change how maj elements are found
    def _find_maj_indices(self, outcomes):
        # Find elements with all 0
        return np.count_nonzero(outcomes, axis = 1) == 0
    
    def _split_outcomes(self, dataset, no_classes):
        if self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
            no_classes = 1

        def __split_outcomes(x, y):
            split_y = tf.split(y, [no_classes, no_classes, no_classes, no_classes, no_classes, no_classes], 1)

            return x, (split_y[0], split_y[1], split_y[2], split_y[3], split_y[4], split_y[5])

        return dataset.map(__split_outcomes, num_parallel_calls=AUTOTUNE)

class joint_narrowing_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None):
        super().__init__(config, 'narrowing_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor)

        self.image_dir = config.train_fixed_location
        self.outcome_columns = ['narrowing_0']
        self.no_classes = 5

    def create_combined_narrowing_joint_dataset(self, outcomes_source, hand_joints_source = './data/predictions/hand_joint_data_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_v2.csv'):
        joint_narrowing_df = self._create_combined_df(outcomes_source, hand_joints_source, feet_joints_source)
        joint_narrowing_dataset = self._create_dream_dataset(joint_narrowing_df, self.outcome_columns, self.no_classes, cache = self.cache)

        return joint_narrowing_dataset

    def _create_combined_df(self, outcomes_source, hand_joints_source, feet_joints_source):
        combined_df = self._create_combined_narrowing_df(hand_joints_source, feet_joints_source)
        combined_outcome_df = self._create_combined_narrowing_outcomes_df(outcomes_source)
        
        return combined_df.merge(combined_outcome_df, on = ['image_name', 'key'])

    def _create_combined_narrowing_df(self, hand_joints_source, feet_joints_source):
        hand_joints_df = self._create_intermediate_joints_df(hand_joints_source, hand_outcome_mapping.keys())
        feet_joints_df = self._create_intermediate_joints_df(feet_joints_source, foot_outcome_mapping.keys())

        return pd.concat([hand_joints_df, feet_joints_df], ignore_index = True, sort = False)

    def _create_combined_narrowing_outcomes_df(self, outcomes_source):
        hand_joints_outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, hand_outcome_mapping, dream_hand_parts)
        feet_joints_outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, foot_outcome_mapping, dream_foot_parts)

        combined_narrowing_outcomes_df =  pd.concat([hand_joints_outcomes_df, feet_joints_outcomes_df], ignore_index=True, sort = False)
        combined_narrowing_outcomes_df = combined_narrowing_outcomes_df.dropna(subset = self.outcome_columns)

        return combined_narrowing_outcomes_df
