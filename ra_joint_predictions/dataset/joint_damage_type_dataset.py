import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from dataset.joint_dataset import dream_dataset
from dataset.test_dataset import joint_test_dataset, combined_test_dataset

hands_joints_source = './data/predictions/hands_joint_data_train_010holdout.csv'
hands_joints_val_source = './data/predictions/hands_joint_data_test_010holdout.csv'

feet_joints_source = './data/predictions/feet_joint_data_train_010holdout.csv'
feet_joints_val_source = './data/predictions/feet_joint_data_test_010holdout.csv'

class joint_damage_type_dataset(dream_dataset):
    def __init__(self, config, pad_resize = False, joint_extractor = None, apply_clahe = False, repeat_test = True):
        super().__init__(config, 'joint_damage_type', pad_resize = pad_resize, joint_extractor = joint_extractor, model_type = "DT")

        self.image_dir = config.train_fixed_location
        self.apply_clahe = apply_clahe
        self.repeat_test = repeat_test
        
        self.n_negatives = 0
        self.n_positives = 0
        self.N = 0
        self.outcomes = np.array([])

    def get_hands_joint_damage_type_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_v2.csv', erosion_flag = False):
        self.cache = self.cache + '_hands'
        
        outcome_column = self._get_outcome_column(erosion_flag)

        return self._get_joint_damage_type_dataset(outcomes_source, joints_source, joint_dataset.hand_outcome_mapping, joint_dataset.dream_hand_parts, [outcome_column])

    def get_hands_joint_damage_type_dataset_with_validation(self, outcomes_source, joints_source = hands_joints_source, joints_val_source = hands_joints_val_source, erosion_flag = False):
        dataset = self.get_hands_joint_damage_type_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)

        test_dataset = self._create_test_dataset()
        val_dataset, val_no_samples = test_dataset.get_hands_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        self.val_outcomes = test_dataset.outcomes
        
        return dataset, val_dataset, val_no_samples

    def get_feet_joint_damage_type_dataset(self, outcomes_source, joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = False):
        self.cache = self.cache + '_feet'
        
        outcome_column = self._get_outcome_column(erosion_flag)

        return self._get_joint_damage_type_dataset(outcomes_source, joints_source, joint_dataset.foot_outcome_mapping, joint_dataset.dream_foot_parts, [outcome_column])

    def get_feet_joint_damage_type_dataset_with_validation(self, outcomes_source, joints_source = feet_joints_source, joints_val_source = feet_joints_val_source, erosion_flag = False):
        dataset = self.get_feet_joint_damage_type_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)

        test_dataset = self._create_test_dataset()
        val_dataset, val_no_samples = test_dataset.get_feet_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        
        self.val_outcomes = test_dataset.outcomes

        return dataset, val_dataset, val_no_samples

    def get_mixed_joint_damage_type_dataset(self, outcomes_source, joint_type, hands_joints_source = './data/predictions/hand_joint_data_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = True):
        self.cache = self.cache + '_mixed'
        
        outcome_column = self._get_outcome_column(erosion_flag)
        
        hand_outcome_joint_df = self._create_joint_damage_type_outcome_df(outcomes_source, hands_joints_source, joint_dataset.hand_outcome_mapping, joint_dataset.dream_hand_parts, [outcome_column])
        feet_outcome_joint_df = self._create_joint_damage_type_outcome_df(outcomes_source, feet_joints_source, joint_dataset.foot_outcome_mapping, joint_dataset.dream_foot_parts, [outcome_column])
        
        self.is_main_hand = joint_type.endswith('H')
        if self.is_main_hand:
            main_dataset, N, n_negatives, outcomes = self._create_joint_damage_type_dataset(hand_outcome_joint_df, [outcome_column], self.cache + '_hands_main')
            sec_dataset, N_sec, n_negatives_sec, _ = self._create_joint_damage_type_dataset(feet_outcome_joint_df, [outcome_column], self.cache + '_feet_sec')
        else:
            main_dataset, N, n_negatives, outcomes = self._create_joint_damage_type_dataset(feet_outcome_joint_df, [outcome_column], self.cache + '_feet_main')
            sec_dataset, N_sec, n_negatives_sec, _ = self._create_joint_damage_type_dataset(hand_outcome_joint_df, [outcome_column], self.cache + '_hands_sec')
            
        # Calc N for step size calculation
        self.N = np.ceil(N * 1.25)
        
        total_N = N + N_sec
        total_n_negatives = n_negatives + n_negatives_sec
        
        self.n_negatives = total_n_negatives
        self.n_positives = total_N - total_n_negatives
        self.alpha = total_n_negatives / total_N
        self.outcomes = np.append(self.outcomes, outcomes)
            
        dataset = tf.data.experimental.sample_from_datasets((main_dataset, sec_dataset), [0.8, 0.2])
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize)
        
        return dataset
    
    def get_mixed_joint_damage_type_dataset_with_validation(self, outcomes_source, joint_type, hands_joints_source = hands_joints_source, hands_joints_val_source = hands_joints_val_source, feet_joints_source = feet_joints_source, feet_joints_val_source = feet_joints_val_source, erosion_flag = False):
        dataset = self.get_mixed_joint_damage_type_dataset(outcomes_source, joint_type, hands_joints_source = hands_joints_source, feet_joints_source = feet_joints_source, erosion_flag = erosion_flag)
        
        if self.is_main_hand:
            test_dataset = self._create_test_dataset()
            val_dataset, val_no_samples = test_dataset.get_hands_joint_test_dataset(joints_source = hands_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        else:
            test_dataset = self._create_test_dataset()
            val_dataset, val_no_samples = test_dataset.get_feet_joint_test_dataset(joints_source = feet_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)
            
        self.val_outcomes = test_dataset.outcomes
            
        return dataset, val_dataset, val_no_samples
        
    def get_combined_joint_damage_type_dataset(self, outcomes_source, hands_joints_source = './data/predictions/hand_joint_data_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = True):
        self.cache = self.cache + '_combined'
        
        outcome_column = self._get_outcome_column(erosion_flag)

        hand_outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, hands_joints_source, joint_dataset.hand_outcome_mapping, joint_dataset.dream_hand_parts)
        feet_outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, feet_joints_source, joint_dataset.foot_outcome_mapping, joint_dataset.dream_foot_parts)

        outcome_joint_df = pd.concat([hand_outcome_joint_df, feet_outcome_joint_df], ignore_index = True, sort = False)
        outcome_joint_df = outcome_joint_df.dropna(subset = [outcome_column])
        
        dataset, N, n_negatives, outcomes = self._create_joint_damage_type_dataset(outcome_joint_df, [outcome_column], self.cache)
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize)
        
        self.N = N
        self.n_negatives = n_negatives
        self.n_positives = N - n_negatives
        self.alpha = n_negatives / N
        self.outcomes = np.append(self.outcomes, outcomes)
        
        return dataset

    def get_combined_joint_damage_type_dataset_with_validation(self, outcomes_source, hands_joints_source = hands_joints_source, hands_joints_val_source = hands_joints_val_source, feet_joints_source = hands_joints_source, feet_joints_val_source = feet_joints_val_source, erosion_flag = False):
        dataset = self.get_combined_joint_damage_type_dataset(outcomes_source, hands_joints_source = hands_joints_source, feet_joints_source = feet_joints_source, erosion_flag = erosion_flag)

        test_dataset = combined_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)
        val_dataset, val_no_samples = test_dataset.get_combined_joint_test_dataset(hand_joints_source = './data/predictions/hand_joint_data_test.csv', feet_joints_source = './data/predictions/feet_joint_data_test.csv', outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        
        self.val_outcomes = test_dataset.outcomes

        return dataset, val_dataset, val_no_samples

    def _get_outcome_column(self, erosion_flag):
        if(erosion_flag):
            outcome_column = 'erosion_0'
            self.cache = self.cache + '_erosion'
        else:
            outcome_column = 'narrowing_0'
            self.cache = self.cache + '_narrowing'

        return outcome_column
    
    def _get_joint_damage_type_dataset(self, outcomes_source, joints_source, outcome_mapping, parts, outcome_columns):
        outcome_joint_df = self._create_joint_damage_type_outcome_df(outcomes_source, joints_source, outcome_mapping, parts, outcome_columns)
        
        dataset, N, n_negatives, outcomes = self._create_joint_damage_type_dataset(outcome_joint_df, outcome_columns, self.cache)
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize)
        
        self.N = N
        self.n_negatives = n_negatives
        self.n_positives = N - n_negatives
        self.alpha = n_negatives / N
        self.outcomes = np.append(self.outcomes, outcomes)
        
        return dataset
    
    def _create_joint_damage_type_outcome_df(self, outcomes_source, joints_source, outcome_mapping, parts, outcome_columns):
        outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, joints_source, outcome_mapping, parts)
        outcome_joint_df = outcome_joint_df.dropna(subset = outcome_columns)
        
        return outcome_joint_df

    def _create_joint_damage_type_dataset(self, outcome_joint_df, outcome_columns, cache):
        outcome_joint_df = outcome_joint_df.sample(frac = 1).reset_index(drop = True)
        
        file_info = outcome_joint_df[['image_name', 'file_type', 'flip', 'key']].to_numpy()

        outcomes = outcome_joint_df[outcome_columns]
        maj_idx = outcomes == 0
        
        N = outcome_joint_df.shape[0]
        n_negatives = np.count_nonzero(maj_idx)
        
        # Set majority samples to 0
        joint_damage_type_outcome = np.ones(file_info.shape[0])
        joint_damage_type_outcome[np.where(maj_idx)[0]] = 0
        
        coords = outcome_joint_df[['coord_x', 'coord_y']].to_numpy()

        return self._create_dataset(file_info, coords, joint_damage_type_outcome, cache), N, n_negatives, outcomes
    
    def _create_outcome_joint_dataframe(self, outcomes_source, joints_source, outcome_mapping, parts):
        outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, outcome_mapping, parts)
        joints_df = self._create_intermediate_joints_df(joints_source, outcome_mapping.keys())

        return outcomes_df.merge(joints_df, on = ['image_name', 'key'])

    def _create_dataset(self, file_info, joint_coords, outcomes, cache):
        dataset = self._create_joint_dataset(file_info, joint_coords, outcomes)
        dataset = self._cache_shuffle_repeat_dataset(dataset, cache, buffer_size = outcomes.shape[0])

        return dataset

    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = 'DT', pad_resize = self.pad_resize, joint_extractor = self.joint_extractor, apply_clahe = self.apply_clahe, repeat = self.repeat_test)
