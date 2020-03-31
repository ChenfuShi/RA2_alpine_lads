import numpy as np
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from dataset.joint_dataset import dream_dataset
from dataset.test_dataset import joint_test_dataset

class joint_damage_type_dataset(dream_dataset):
    def __init__(self, config, pad_resize = False, joint_extractor = None, alpha = 0.75):
        super().__init__(config, 'joint_damage_type', pad_resize = pad_resize, joint_extractor = joint_extractor, model_type = "DT")

        self.image_dir = config.train_fixed_location
        self.alpha = alpha

    def get_hands_joint_damage_type_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_v2.csv', erosion_flag = False):
        outcome_column = self._get_outcome_column(erosion_flag)

        return self._create_joint_damage_dataset(outcomes_source, joints_source, joint_dataset.hand_outcome_mapping, joint_dataset.dream_hand_parts, [outcome_column])

    def get_hands_joint_damage_type_dataset_with_validation(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_train_v2.csv', joints_val_source = './data/predictions/hand_joint_data_test_v2.csv', erosion_flag = False):
        dataset = self.get_hands_joint_damage_type_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)

        val_dataset, val_no_samples = self._create_test_dataset().get_hands_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples

    def get_feet_joint_damage_type_dataset(self, outcomes_source, joints_source = './data/predictions/feet_joint_data_v2.csv', erosion_flag = False):
        outcome_column = self._get_outcome_column(erosion_flag)

        return self._create_joint_damage_dataset(outcomes_source, joints_source, joint_dataset.foot_outcome_mapping, joint_dataset.dream_foot_parts, [outcome_column])

    def get_feet_joint_damage_type_dataset_with_validation(self, outcomes_source, joints_source = './data/predictions/feet_joint_data_train_v2.csv', joints_val_source = './data/predictions/feet_joint_data_test_v2.csv', erosion_flag = False):
        dataset = self.get_feet_joint_damage_type_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)

        val_dataset, val_no_samples = self._create_test_dataset().get_feet_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples

    def _get_outcome_column(self, erosion_flag):
        if(erosion_flag):
            outcome_column = 'erosion_0'
            self.cache = self.cache + '_erosion'
        else:
            outcome_column = 'narrowing_0'
            self.cache = self.cache + '_narrowing'

        return outcome_column

    def _create_joint_damage_dataset(self, outcomes_source, joints_source, outcome_mapping, parts, outcome_columns):
        outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, joints_source, outcome_mapping, parts)
        outcome_joint_df = outcome_joint_df.dropna(subset = outcome_columns)

        dataset = self._create_joint_damage_type_dataset(outcome_joint_df, outcome_columns)
        
        return dataset

    def _create_outcome_joint_dataframe(self, outcomes_source, joints_source, outcome_mapping, parts):
        outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, outcome_mapping, parts)
        joints_df = self._create_intermediate_joints_df(joints_source, outcome_mapping.keys())

        return outcomes_df.merge(joints_df, on = ['image_name', 'key'])

    def _create_joint_damage_type_dataset(self, outcome_joint_df, outcome_column):
        outcome_joint_df = outcome_joint_df.sample(frac = 1).reset_index(drop = True)

        file_info = outcome_joint_df[['image_name', 'file_type', 'flip', 'key']].to_numpy()

        outcomes = outcome_joint_df[outcome_column]
        maj_idx = outcomes == 0
        
        # Set majority samples to 0
        joint_damage_type_outcome = np.ones(file_info.shape[0])
        joint_damage_type_outcome[np.where(maj_idx)[0]] = 0
        
        coords = outcome_joint_df[['coord_x', 'coord_y']].to_numpy()

        return self._create_non_split_joint_dataset(file_info, coords, joint_damage_type_outcome, augment = True, cache = self.cache, buffer_size = 2000)

    def _create_dataset(self, file_info, joint_coords, outcomes, maj_idx):
        min_idx = np.logical_not(maj_idx)

        # Tranform boolean mask into indices
        maj_idx = np.where(maj_idx)[0]
        min_idx = np.where(min_idx)[0]

        # Create 2 datasets, one with the majority class, one with the other classes
        maj_ds = self._create_joint_dataset(file_info[maj_idx, :], joint_coords[maj_idx], outcomes[maj_idx])
        min_ds = self._create_joint_dataset(file_info[min_idx, :], joint_coords[min_idx], outcomes[min_idx])

        # Cache the partial datasets, shuffle the datasets with buffersize that ensures minority samples are all shuffled
        maj_ds = self._cache_shuffle_repeat_dataset(maj_ds, self.cache + '_maj', buffer_size = min_idx.shape[0])
        min_ds = self._cache_shuffle_repeat_dataset(min_ds, self.cache + '_min', buffer_size = min_idx.shape[0])

        # Interleave datasets, inverse of alpha (if we want alpha to 0.75, then we want the minority to sample to be only 25% of samples)
        dataset = tf.data.experimental.sample_from_datasets((maj_ds, min_ds), [1 - self.alpha, self.alpha])

        return self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize)

    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = 'DT', pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)