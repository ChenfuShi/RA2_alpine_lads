import logging

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from model.joint_damage_model import load_joint_damage_model
from model.utils.metrics import argmax_rmse, softmax_rmse_metric, class_softmax_rmse_metric
from dataset.test_dataset import joint_test_dataset

def predict_dream_test_set(config, predict_params):
    hands_narrowing_model, wrists_narrowing_model, feet_narrowing_model, hands_erosion_model, wrists_erosion_model, feet_erosion_model = _get_models(predict_params)
    
    preds = {}
    
    def _init_preds(patient_id):
        if not patient_id in preds.keys():
            preds[patient_id] = {
                'Patient_ID': patient_id
        }

    def _predict(test_dataset, outcome_mapping, narrowing_model, erosion_model, no_narrowing_classes = 5, no_erosion_classes = 6):
        for file_info, img in test_dataset:
            patient_id, part, key = _get_details(file_info)
        
            _init_preds(patient_id)
        
            img_tf = tf.expand_dims(img, 0)
            
            y_preds = narrowing_model.predict(img_tf)
            
            narrowing_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][0]]
            for idx, narrowing_outcome_key in enumerate(narrowing_outcome_keys):
                y_pred = y_preds[idx]
                y_pred = np.sum(y_pred * np.arange(no_narrowing_classes))
                
                preds[patient_id][narrowing_outcome_key] = y_pred

            y_preds = erosion_model.predict(img_tf)
            
            erosion_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][1]]
            for idx, erosion_outcome_key in enumerate(erosion_outcome_keys):
                y_pred = y_preds[idx]
                y_pred = np.sum(y_pred * np.arange(no_erosion_classes))
                
                preds[patient_id][erosion_outcome_key] = y_pred

    dataset = joint_test_dataset(config, config.train_fixed_location)

    hands_test_dataset, _ = dataset.get_hands_joint_test_dataset(joints_source = predict_params['hands_joint_source'])
    wrists_test_dataset, _ = dataset.get_wrists_joint_test_dataset(joints_source = predict_params['hands_joint_source'])
    feet_test_dataset, _ = dataset.get_feet_joint_test_dataset(joints_source = predict_params['feet_joint_source'])
  
    _predict(hands_test_dataset, joint_dataset.hand_outcome_mapping, hands_narrowing_model, hands_erosion_model, no_narrowing_classes = 5, no_erosion_classes = 6)
    _predict(wrists_test_dataset, joint_dataset.wrist_outcome_mapping, wrists_narrowing_model, wrists_erosion_model, no_narrowing_classes = 5, no_erosion_classes = 6)
    _predict(feet_test_dataset, joint_dataset.foot_outcome_mapping, feet_narrowing_model, feet_erosion_model, no_narrowing_classes = 5, no_erosion_classes = 11)

    predictions_df = pd.DataFrame(preds.values(), index = np.arange(len(preds.values())))
    
    narrowing_mask = ['_J_' in column_name for column_name in predictions_df.columns]
    erosion_mask = ['_E_' in column_name for column_name in predictions_df.columns]

    narrowing_sum = np.sum(predictions_df.iloc[:, narrowing_mask].to_numpy(), axis = 1)
    erosion_sum = np.sum(predictions_df.iloc[:, erosion_mask].to_numpy(), axis = 1)
    
    total_sum = narrowing_sum + erosion_sum
    
    predictions_df['Overall_narrowing'] = narrowing_sum
    predictions_df['Overall_erosion'] = erosion_sum
    predictions_df['Overall_Tol'] = total_sum
    
    template = pd.read_csv(predict_params['template_path'])
    predictions_df = predictions_df[template.columns]
    
    predictions_df.to_csv(predict_params['output_path'], index = False)
            
def _get_models(predict_params):
    hands_narrowing_model = load_joint_damage_model(predict_params['hands_narrowing_model'], 5)
    wrists_narrowing_model = load_joint_damage_model(predict_params['wrists_narrowing_model'], 5)
    feet_narrowing_model = load_joint_damage_model(predict_params['feet_narrowing_model'], 5)

    hands_erosion_model = load_joint_damage_model(predict_params['hands_erosion_model'], 6)
    wrists_erosion_model = load_joint_damage_model(predict_params['wrists_erosion_model'], 6)
    feet_erosion_model = load_joint_damage_model(predict_params['feet_erosion_model'], 11)

    return hands_narrowing_model, wrists_narrowing_model, feet_narrowing_model, hands_erosion_model, wrists_erosion_model, feet_erosion_model
                       
def _get_details(file_info):
    file_info = file_info.numpy()
    
    img_info = file_info[0].decode('utf-8').split('.')[0]
    key = file_info[3].decode('utf-8')
    
    patient_info = img_info.split('-')
    patient_id = patient_info[0]
    part = patient_info[1]
        
    return patient_id, part, key
