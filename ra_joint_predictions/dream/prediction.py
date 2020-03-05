import logging

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from model.utils.metrics import argmax_rsme, softmax_rsme_metric, class_softmax_rsme_metric
from dataset.test_dataset import joint_test_dataset

def predict_dream_test_set(config):
    hands_narrowing_model, wrists_narrowing_model, feet_narrowing_model, hands_erosion_model, wrists_erosion_model, feet_erosion_model = _get_models()
    
    preds = {}
    
    def _init_preds(patient_id):
        if not patient_id in preds.keys():
            preds[patient_id] = {
                'Patient_ID': patient_id
        }
            
    dataset = joint_test_dataset(config, config.train_fixed_location)

    hands_test_dataset = dataset.get_hand_joint_test_dataset(joints_source = '/output/dream_test_hand_joint_data.csv')
    wrists_test_dataset = dataset.get_wrist_joint_testdataset(joints_source = '/output/dream_test_hand_joint_data.csv')
    feet_test_dataset = dataset.get_feet_joint_test_dataset(joints_source = '/output/dream_test_feet_joint_data.csv')
  
    for file_info, img in hands_test_dataset:
        patient_id, part, key = _get_details(file_info)
        
        _init_preds(patient_id)
        
        img_tf = tf.expand_dims(img, 0)
        
        outcomes = joint_dataset.hand_outcome_mapping[key][0]
        if len(outcomes) > 0:
            narrowing_outcome_key = outcomes[0].format(part = part)
            y_pred = hands_narrowing_model.predict(img_tf)
            y_pred = np.sum(y_pred * np.arange(5), axis = 1)
            preds[patient_id][narrowing_outcome_key] = y_pred[0]
        
        outcomes = joint_dataset.hand_outcome_mapping[key][1]
        if len(outcomes) > 0:
            erosion_outcome_key = outcomes[0].format(part = part)
            y_pred = hands_erosion_model.predict(img_tf)
            y_pred = np.sum(y_pred * np.arange(6), axis = 1)
            preds[patient_id][erosion_outcome_key] = y_pred[0]
        
    for file_info, img in wrists_test_dataset:
        patient_id, part, key = _get_details(file_info)
        
        _init_preds(patient_id)
        
        img_tf = tf.expand_dims(img, 0)
        
        wrist_narrowing_outcome_keys = [wrist_key.format(part = part) for wrist_key in joint_dataset.wrist_outcome_mapping['wrist'][0]]
        wrist_erosion_outcome_keys = [wrist_key.format(part = part) for wrist_key in joint_dataset.wrist_outcome_mapping['wrist'][1]]
        
        y_pred = wrists_narrowing_model.predict(img_tf)
        for idx, narrowing_key in enumerate(wrist_narrowing_outcome_keys):
            preds[patient_id][narrowing_key] = np.sum(y_pred[idx] * np.arange(5), axis = 1)[0]
            
        y_pred = wrists_erosion_model.predict(img_tf)
        for idx, erosion_key in enumerate(wrist_erosion_outcome_keys):
            preds[patient_id][erosion_key] = np.sum(y_pred[idx] * np.arange(6), axis = 1)[0]
    
    for file_info, img in feet_test_dataset:
        patient_id, part, key = _get_details(file_info)
        
        _init_preds(patient_id)
        
        img_tf = tf.expand_dims(img, 0)
        
        narrowing_outcome_key = joint_dataset.foot_outcome_mapping[key][0][0].format(part = part)
        y_pred = feet_narrowing_model.predict(img_tf)
        y_pred = np.sum(y_pred * np.arange(5), axis = 1)
        preds[patient_id][narrowing_outcome_key] = y_pred[0]
        
        erosion_outcome_key = joint_dataset.foot_outcome_mapping[key][1][0].format(part = part)
        y_pred = feet_erosion_model.predict(img_tf)
        y_pred = np.sum(y_pred * np.arange(11), axis = 1)
        preds[patient_id][erosion_outcome_key] = y_pred[0]
        
    predictions_df = pd.DataFrame(preds.values(), index = np.arange(len(preds.values())))
    
    narrowing_mask = ['_J_' in column_name for column_name in predictions_df.columns]
    erosion_mask = ['_E_' in column_name for column_name in predictions_df.columns]
    
    narrowing_sum = np.sum(predictions_df.iloc[:, narrowing_mask].to_numpy(), axis = 1)
    erosion_sum = np.sum(predictions_df.iloc[:, erosion_mask].to_numpy(), axis = 1)
    
    total_sum = narrowing_sum + erosion_sum
    
    predictions_df['Overall_narrowing'] = narrowing_sum
    predictions_df['Overall_erosion'] = erosion_sum
    predictions_df['Overall_Tol'] = total_sum
    
    template = pd.read_csv('/test/template.csv')
    predictions_df = predictions_df[template.columns]
    
    predictions_df.to_csv('/output/predictions.csv', index = False)
            
def _get_models():
    dependencies = {
        'softmax_rsme': softmax_rsme_metric(np.arange(5)),
        'argmax_rsme': argmax_rsme,
        'class_softmax_rmse_0': class_softmax_rsme_metric(np.arange(5), 0)
    }

    feet_narrowing_model = tf.keras.models.load_model('./pretrained_models/feet_narrowing_v1.h5', custom_objects=dependencies)
    hands_narrowing_model = tf.keras.models.load_model('./pretraiend_models/hands_narrowing_v1.h5', custom_objects=dependencies)
    wrists_narrowing_model = tf.keras.models.load_model('./pretrained_models/wrists_narrowing_v2.h5', custom_objects=dependencies)
    
    dependencies = {
        'softmax_rsme': softmax_rsme_metric(np.arange(6)),
        'argmax_rsme': argmax_rsme,
        'class_softmax_rmse_0': class_softmax_rsme_metric(np.arange(6), 0)
    }

    hands_erosion_model = tf.keras.models.load_model('./pretrained_models/hands_erosion_v1.h5', custom_objects=dependencies)
    wrists_erosion_model = tf.keras.models.load_model('./pretrained_models/wrists_erosion_v2.h5', custom_objects=dependencies)
    
    dependencies = {
        'softmax_rsme': softmax_rsme_metric(np.arange(11)),
        'argmax_rsme': argmax_rsme,
        'class_softmax_rmse_0': class_softmax_rsme_metric(np.arange(11), 0)
    }
                       
    feet_erosion_model = tf.keras.models.load_model('./pretrained_models/feet_erosion_v2.h5', custom_objects=dependencies)

    return hands_narrowing_model, wrists_narrowing_model, feet_narrowing_model, hands_erosion_model, wrists_erosion_model, feet_erosion_model
                       
def _get_details(file_info):
    file_info = file_info.numpy()
    
    img_info = file_info[0].decode('utf-8').split('.')[0]
    key = file_info[3].decode('utf-8')
    
    patient_info = img_info.split('-')
    patient_id = patient_info[0]
    part = patient_info[1]
        
    return patient_id, part, key
