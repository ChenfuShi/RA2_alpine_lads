import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.dataset_ops as ops

def get_landmarks_dataset(landmarks_location):
    landmarks_dataframe = create_landmarks_dataframe(landmarks_location)
    
    label_idx = np.logical_and(landmarks_dataframe.columns != 'sample_id', landmarks_dataframe.columns != 'flip')
    
    landmarks_dataset = tf.data.Dataset.from_tensor_slices((landmarks_dataframe[['sample_id', 'flip']].values, landmarks_dataframe.loc[:, label_idx].values))
    landmarks_dataset = ops.load_images(landmarks_dataset, 'C:\\Users\\CrankMuffler\\Development\\Dream\\training_v2020_01_13', True, 'RF')
    
    return landmarks_dataset

def create_landmarks_dataframe(landmarks_location):
    landmark_files = os.listdir(landmarks_location)
    
    df_rows = []
    for landmark_file in landmark_files:
        landmark_file_path = os.path.join(landmarks_location, landmark_file)
        
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
    
    N_points = len(sorted_shapes)
    
    labels_dict = {}
    
    for idx, shape in enumerate(sorted_shapes):
        key = shape['label']
        
        labels_dict[key + '_x'] = shape['points'][0][0]
        labels_dict[key + '_y'] = shape['points'][0][1]
        
    return labels_dict