import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.base_dataset import dream_dataset
import model.joint_damage_model as joint_damage_model

from tensorflow.python.data.ops import dataset_ops

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

class feet_joint_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None, divide_erosion_by_2 = False):
        super().__init__(config, 'feet_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type)

        self.image_dir = config.train_fixed_location
        
        self.maj_ratio = 0.25
        self.divide_erosion_by_2 = divide_erosion_by_2

    def create_feet_joints_dataset(self, outcomes_source, joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = False):
        self.erosion_flag = erosion_flag
        
        outcome_column = 'narrowing_0'
        no_classes = 5
        self.cache = self.cache + '_narrowing'
        if(erosion_flag):
            outcome_column = 'erosion_0'
            no_classes = 11
            self.cache = self.cache + '_erosion'

        return self._create_dream_datasets(outcomes_source, joints_source, foot_outcome_mapping, dream_foot_parts, [outcome_column], no_classes)
    
    def _get_idx_groups(self, outcomes):
        if self.erosion_flag:
            if not self.divide_erosion_by_2:
                idx_groups = [outcomes == 0, np.logical_or(outcomes == 1, outcomes == 2), np.logical_or(outcomes == 3, outcomes == 4), np.logical_or(outcomes == 5, outcomes == 6), outcomes >= 7]
            else:
                idx_groups = [outcomes == 0, np.logical_or(outcomes == 0.5, outcomes == 1), np.logical_or(outcomes == 1.5, outcomes == 2), np.logical_or(outcomes == 2.5, outcomes == 3), outcomes >= 3.5]
        else:
            idx_groups = [outcomes == 0, outcomes == 1, outcomes == 2, outcomes == 3, outcomes == 4]
        
        return idx_groups
    
    def _get_outcomes(self, outcomes, no_classes):
        tf_dummy_outcomes, tf_outcomes = super()._get_outcomes(outcomes, no_classes)
        
        if self.erosion_flag and self.divide_erosion_by_2:
            tf_outcomes = tf_outcomes / 2
            
        return tf_dummy_outcomes, tf_outcomes
    
class hands_joints_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None, apply_clahe = False, multiply_by_two = False):
        super().__init__(config, 'hands_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type)

        self.image_dir = config.train_fixed_location
        self.apply_clahe = apply_clahe
        
        self.maj_ratio = 0.25
        self.multiply_erosion_by_2 = multiply_by_two
        
    def create_hands_joints_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_v2.csv', erosion_flag = False):
        self.erosion_flag = erosion_flag
        
        outcome_column = 'narrowing_0'
        no_classes = 5
        self.cache = self.cache + '_narrow'
        if(erosion_flag):
            outcome_column = 'erosion_0'
            no_classes = 6
            self.cache = self.cache + '_erosion'

        return self._create_dream_datasets(outcomes_source, joints_source, hand_outcome_mapping, dream_hand_parts, [outcome_column], no_classes)
    
    def _get_idx_groups(self, outcomes):
        if self.model_type == 'C':
            outcomes = np.argmax(outcomes, axis=1)

        if self.erosion_flag:
            if not self.multiply_erosion_by_2:
                idx_groups = [outcomes == 0, outcomes == 1, outcomes == 2, outcomes == 3, outcomes >= 4]
            else:
                idx_groups = [outcomes == 0, outcomes == 2, outcomes == 4, outcomes == 6, outcomes >= 8]
        else:
            idx_groups = [outcomes == 0, outcomes == 1, outcomes == 2, outcomes == 3, outcomes == 4]
            
        return idx_groups
    
    def _get_outcomes(self, outcomes, no_classes):
        tf_dummy_outcomes, tf_outcomes = super()._get_outcomes(outcomes, no_classes)
        
        if self.erosion_flag and self.multiply_erosion_by_2:
            tf_outcomes = tf_outcomes * 2
            
        return tf_dummy_outcomes, tf_outcomes

class hands_wrists_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, 'wrists_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location
        self.is_wrist = True
        self.maj_ratio = 0.5

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
    
class mixed_joint_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None, joint_type = 'HF'):
        super().__init__(config, 'mixed_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)
        
        self.joint_type = joint_type
        self.is_main_hand = joint_type.endswith('H')
        
        if self.is_main_hand:
            self.main_ds = hands_joints_dataset(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type)
            self.sec_ds = feet_joint_dataset(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type, divide_erosion_by_2 = True)
        else:
            self.main_ds = feet_joint_dataset(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type)
            self.sec_ds = hands_joints_dataset(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type, multiply_by_two = True)
            
        self.maj_ratio = 0.8
        
    def create_mixed_joint_dataset(self, outcomes_source, hand_joints_source = './data/predictions/hand_joint_data_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = False):
        if self.is_main_hand:
            main_ds = self.main_ds.create_hands_joints_dataset(outcomes_source, joints_source = hand_joints_source, erosion_flag = erosion_flag)
            sec_ds = self.sec_ds.create_feet_joints_dataset(outcomes_source, joints_source = feet_joints_source, erosion_flag = erosion_flag) 
        else:
            main_ds = self.main_ds.create_feet_joints_dataset(outcomes_source, joints_source = feet_joints_source, erosion_flag = erosion_flag)
            sec_ds = self.sec_ds.create_hands_joints_dataset(outcomes_source, joints_source = hand_joints_source, erosion_flag = erosion_flag)
            
        main_ds = main_ds.unbatch()
        sec_ds = sec_ds.unbatch()
        
        dataset = tf.data.experimental.sample_from_datasets((main_ds, sec_ds), [self.maj_ratio, 1 - self.maj_ratio]) 
        
        dataset = dataset.batch(64)
        
        self.class_weights = self.main_ds.class_weights
        
        return dataset

class combined_joint_dataset(dream_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None):
        super().__init__(config, 'combined_joints', model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location
        self.no_classes = 5
        
        self.hands_dataset = hands_joints_dataset(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = 'balanced')
        self.feet_dataset = feet_joint_dataset(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type)

    def create_combined_joint_dataset(self, outcomes_source, hand_joints_source = './data/predictions/hand_joint_data_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = False):
        if erosion_flag:
            self.outcome_columns = ['erosion_0']
            self.cache = self.cache + '_erosion'

            if self.model_type != 'DT':
                logging.warn('Combined dataset for erosion only supports model_type DT!')
        else:
            self.outcome_columns = ['narrowing_0']
            self.cache = self.cache + '_narrowing'
           
        combined_joint_df = self._create_combined_df(outcomes_source, hand_joints_source, feet_joints_source)
        dataset = self._create_dream_dataset(combined_joint_df, self.outcome_columns, self.no_classes, cache = self.cache)
        
        return dataset

    def _create_combined_df(self, outcomes_source, hand_joints_source, feet_joints_source):
        combined_df = self._create_combined_joints_df(hand_joints_source, feet_joints_source)
        combined_outcome_df = self._create_combined_outcomes_df(outcomes_source)
        
        return combined_df.merge(combined_outcome_df, on = ['image_name', 'key'])

    def _create_combined_joints_df(self, hand_joints_source, feet_joints_source):
        hand_joints_df = self._create_intermediate_joints_df(hand_joints_source, hand_outcome_mapping.keys())
        feet_joints_df = self._create_intermediate_joints_df(feet_joints_source, foot_outcome_mapping.keys())

        return pd.concat([hand_joints_df, feet_joints_df], ignore_index = True, sort = False)

    def _create_combined_outcomes_df(self, outcomes_source):
        hand_joints_outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, hand_outcome_mapping, dream_hand_parts)
        feet_joints_outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, foot_outcome_mapping, dream_foot_parts)

        combined_outcomes_df =  pd.concat([hand_joints_outcomes_df, feet_joints_outcomes_df], ignore_index = True, sort = False)
        combined_outcomes_df = combined_outcomes_df.dropna(subset = self.outcome_columns)

        return combined_outcomes_df
    
    def _get_idx_groups(self, outcomes):
        idx_groups = [outcomes == 0, outcomes == 1, outcomes == 2, outcomes == 3, outcomes == 4]
        
        return idx_groups
