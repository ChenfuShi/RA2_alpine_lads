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

    def _create_dataset(self, x, y, z, val_split = False):
        dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
        dataset = joint_ops.load_joints(dataset, self.config.train_location)

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

    def create_feet_joints_dataset(self, narrowing_flag, joint_source = './data/feet_joint_data.csv', val_split = False):
        feet_dataframe = pd.read_csv(joint_source)

        feet_dataframe['flip'] = 'N'
        flip_idx = ['-R' in image_name for image_name in feet_dataframe['image_name']]
        flip_columns = feet_dataframe['flip'].values
        flip_columns[flip_idx] = 'Y'
        feet_dataframe['flip'] = flip_columns

        feet_dataframe['file_type'] = 'jpg'

        x = feet_dataframe[['image_name', 'file_type', 'flip', 'key']].values
        
        outcome_column = 'erosion_0'
        if(narrowing_flag):
            outcome_column = 'narrowing_0'

        outcome = OneHotEncoder(categories='auto', sparse = False).fit_transform(feet_dataframe[outcome_column].values.reshape((-1, 1)) - 1)

        y = feet_dataframe[['coord_x', 'coord_y']].values

        return self._create_dataset(x, y, outcome, val_split = val_split)