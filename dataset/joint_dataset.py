import pandas as pd
import tensorflow as tf

import dataset.dataset_ops as ops
import dataset.ops.joint_ops as joint_ops
import dataset.image_ops as img_ops

from dataset.base_dataset import base_dataset

class joint_dataset(base_dataset):
    def __init__(self, config):
        super().__init__(config)

    def create_joints_dataset(self):
        dataframe = pd.read_csv('thing.csv')

        x = dataframe[['image_name', 'flip', 'key']].values
        y = dataframe[['coord_x', 'coord_y', 'erosion_0']].values

        dataset = self._create_dataset(x, y, self.config.train_location, update_labels = False)
        dataset = self._prepare_for_training(dataset, 256, 128, batch_size = 16, augment = False)

        return dataset

    def _create_dataset(self, x, y, file_location, update_labels = False):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = joint_ops.load_joints(dataset, file_location)

        return dataset
