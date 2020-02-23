import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.base_dataset import base_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class landmarks_dataset(base_dataset):
    def __init__(self, config):
        super().__init__(config)
        
    def _create_landmarks_dataset(self, image_location, landmarks_location, create_val, shuffle_before_val = False):
        landmarks_dataframe = _create_landmarks_dataframe(landmarks_location)
        
        self.landmarks_dataframe = landmarks_dataframe
        
        label_idx = np.logical_and(landmarks_dataframe.columns != 'file_type', np.logical_and(landmarks_dataframe.columns != 'sample_id', landmarks_dataframe.columns != 'flip'))
        x = landmarks_dataframe[['sample_id', 'file_type', 'flip']].values
        y = landmarks_dataframe.loc[:, label_idx].values
        
        dataset = super()._create_dataset(x, y, image_location, update_labels = True)

        if create_val:
            if shuffle_before_val:
                dataset = dataset.shuffle(buffer_size = 700,seed=65)
            dataset, val_dataset = super()._create_validation_split(dataset)

        dataset = super()._prepare_for_training(dataset, self.config.landmarks_img_width, self.config.landmarks_img_height, batch_size = self.config.batch_size, update_labels = True)

        if create_val:
            val_dataset = super()._prepare_for_training(val_dataset, self.config.landmarks_img_width, self.config.landmarks_img_height, batch_size = self.config.batch_size, update_labels = True, augment = False)

            return dataset, val_dataset

        return dataset

class feet_landmarks_dataset(landmarks_dataset):
    def __init__(self, config):
        super().__init__(config)

    def create_landmarks_dataset(self, create_val = False, shuffle_before_val = False):
        image_location = os.path.join(self.config.landmarks_feet_images_location , self.config.fixed_dir)
        landmarks_location = os.path.join(self.config.landmarks_location, 'feet')

        return self._create_landmarks_dataset(image_location, landmarks_location, create_val, shuffle_before_val)

class hands_landmarks_dataset(landmarks_dataset):
    def __init__(self, config):
        super().__init__(config)

    def create_landmarks_dataset(self, create_val = False, shuffle_before_val = False):
        image_location = os.path.join(self.config.landmarks_hands_images_location , self.config.fixed_dir)
        landmarks_location = os.path.join(self.config.landmarks_location, 'hands')

        return self._create_landmarks_dataset(image_location, landmarks_location, create_val, shuffle_before_val)


def _create_landmarks_dataframe(landmarks_location):
    landmark_files = os.listdir(landmarks_location)
    
    df_rows = []
    for landmark_file in landmark_files:
        landmark_file_path = os.path.join(landmarks_location, landmark_file)
        
        if(landmark_file_path.endswith('.json')):
            with open(landmark_file_path) as landmark_json_file:
                landmark_json = json.load(landmark_json_file)
                
                labels_dict = _get_labels_for_landmarks_json(landmark_json)
                
                sample_id = landmark_file.split('.')[0]

                # last element of the stored img_path is the file type
                img_path = landmark_json['imagePath'].split('.')
                file_type = img_path[-1]

                labels_dict['sample_id'] = sample_id
                labels_dict['file_type'] = "jpg" #### hard coding jpg
                
                if 'RF' in sample_id or 'RH' in sample_id:
                    labels_dict['flip'] = 'Y'
                else:
                    labels_dict['flip'] = 'N'
                
                df_rows.append(labels_dict)

    return pd.DataFrame(df_rows, index = np.arange(len(df_rows)))
    
def _get_labels_for_landmarks_json(landmarks_json):
    sorted_shapes = sorted(landmarks_json['shapes'], key = lambda k: k['label']) 
    
    labels_dict = {}
    for shape in sorted_shapes:
        key = shape['label']
        
        labels_dict[key + '_x'] = shape['points'][0][0]
        labels_dict[key + '_y'] = shape['points'][0][1]
        
    return labels_dict