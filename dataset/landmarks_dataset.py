import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.dataset_ops as ops
import dataset.image_ops as img_ops

from dataset.base_dataset import base_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class landmarks_dataset(base_dataset):
    def __init__(self, config):
        super().__init__(config)
        
    def create_landmarks_dataset(self):
        landmarks_dataframe = _create_landmarks_dataframe(self.config.landmarks_location)
        
        self.landmarks_dataframe = landmarks_dataframe
        
        label_idx = np.logical_and(landmarks_dataframe.columns != 'sample_id', landmarks_dataframe.columns != 'flip')
        x = landmarks_dataframe[['sample_id', 'flip']].values
        y = landmarks_dataframe.loc[:, label_idx].values
        
        dataset = super()._create_dataset(x, y, self.config.train_location, update_labels = True)

        return super()._prepare_for_training(dataset, self.config.landmarks_img_width, self.config.landmarks_img_height, batch_size = self.config.batch_size, cache = self.config.cache_loc + 'landmarks_cache', update_labels = True)

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
                labels_dict['sample_id'] = sample_id
                
                if 'RF' in sample_id:
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