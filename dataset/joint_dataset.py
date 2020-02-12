import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

import dataset.ops.joint_ops as joint_ops
import dataset.ops.dataset_ops as ds_ops

from dataset.base_dataset import base_dataset

class joint_dataset(base_dataset):
    def __init__(self, config, cache_postfix):
        super().__init__(config)

        self.cache = config.cache_loc + cache_postfix
        self.image_dir = config.train_location

    def _create_dataset(self, file_info, joint_coords, outcomes, val_split = False):
        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords, outcomes))
        dataset = joint_ops.load_joints(dataset, self.image_dir)

        if val_split:
            dataset, val_dataset = self._create_validation_split(dataset, split_size = 200)

        dataset = self._prepare_for_training(dataset, 256, 128, batch_size = self.config.batch_size, cache = self.cache, pad_resize = False)

        if val_split:
            val_dataset = self._prepare_for_training(val_dataset, 256, 128, batch_size = self.config.batch_size, pad_resize = False, augment = False)

            return dataset, val_dataset
        else:
            return dataset

class feet_joint_dataset(joint_dataset):
    def __init__(self, config):
        super().__init__(config, 'feet_joints')

        self.image_dir = config.train_location

    def create_feet_joints_dataset(self, narrowing_flag, joint_source = './data/feet_joint_data.csv', val_split = False):
        feet_dataframe = pd.read_csv(joint_source)

        feet_dataframe['flip'] = 'N'
        flip_idx = ['-R' in image_name for image_name in feet_dataframe['image_name']]
        flip_columns = feet_dataframe['flip'].values
        flip_columns[flip_idx] = 'Y'
        feet_dataframe['flip'] = flip_columns

        feet_dataframe['file_type'] = 'jpg'

        file_info = feet_dataframe[['image_name', 'file_type', 'flip', 'key']].values
        
        outcome_column = 'erosion_0'
        if(narrowing_flag):
            outcome_column = 'narrowing_0'

        outcomes = OneHotEncoder(categories='auto', sparse = False).fit_transform(feet_dataframe[outcome_column].values.reshape((-1, 1)) - 1)

        coords = feet_dataframe[['coord_x', 'coord_y']].values

        return self._create_dataset(file_info, coords, outcomes, val_split = val_split)

class rsna_joint_dataset(joint_dataset):
    def __init__(self, config):
        super().__init__(config, 'rsna_joints')

        self.image_dir = '../rsna_boneAge/checked_rsna_training'

    def create_rsna_joints_dataset(self, joint_source = './data/rsna_joint_data.csv', val_split = False):
        joint_dataframe = pd.read_csv(joint_source)

        joint_dataframe['flip'] = 'N'
        joint_dataframe['file_type'] = 'png'

        file_info = joint_dataframe[['image_name', 'file_type', 'flip', 'key']].astype(np.str).values
        coords = joint_dataframe[['coord_x', 'coord_y']].values

        outcomes = joint_dataframe[['boneage', 'sex', 'key']]
        outcomes = pd.get_dummies(outcomes, columns = ['key']).values

        return self._create_dataset(file_info, coords, outcomes, val_split = val_split) 