import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from dataset.test_dataset import joint_test_dataset
from prediction.joint_damage_prediction import joint_damage_predictor

def predict_test_set(config, model_parameters_collection, hands_joint_source = './data/predictions/hand_joint_data_test_v2.csv', feet_joint_source = './data/predictions/feet_joint_data_test_v2.csv'):
    hands_test_dataset, wrists_test_dataset, feet_test_dataset = _get_test_datasets(config, hands_joint_source, feet_joint_source)
    hand_narrowing_predictor, wrists_narrowing_predictor, feet_narrowing_predictor, hand_erosion_predictor, wrists_erosion_predictor, feet_erosion_predictor = _get_joint_damage_predictors(model_parameters_collection)
    
    preds = {}
    
    def _init_preds(patient_id):
        if not patient_id in preds.keys():
            preds[patient_id] = {
                'Patient_ID': patient_id
        }

    def _predict(test_dataset, outcome_mapping, narrowing_predictor, erosion_predictor):
        for file_info, img in test_dataset:
            patient_id, part, key = _get_details(file_info)
        
            _init_preds(patient_id)
        
            img_tf = tf.expand_dims(img, 0)
            
            y_preds = narrowing_predictor.predict_joint_damage(img_tf)
            narrowing_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][0]]
            for idx, narrowing_outcome_key in enumerate(narrowing_outcome_keys):
                y_pred = y_preds[idx]
                
                preds[patient_id][narrowing_outcome_key] = y_pred

            y_preds = erosion_predictor.predict_joint_damage(img_tf)
            erosion_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][1]]
            for idx, erosion_outcome_key in enumerate(erosion_outcome_keys):
                y_pred = y_preds[idx]
                
                preds[patient_id][erosion_outcome_key] = y_pred

    _predict(hands_test_dataset, joint_dataset.hand_outcome_mapping, hand_narrowing_predictor, hand_erosion_predictor)
    _predict(wrists_test_dataset, joint_dataset.wrist_outcome_mapping, wrists_narrowing_predictor, wrists_erosion_predictor)
    _predict(feet_test_dataset, joint_dataset.foot_outcome_mapping, feet_narrowing_predictor, feet_erosion_predictor)

    predictions_df = pd.DataFrame(preds.values(), index = np.arange(len(preds.values())))
    
    narrowing_mask = ['_J_' in column_name for column_name in predictions_df.columns]
    erosion_mask = ['_E_' in column_name for column_name in predictions_df.columns]

    narrowing_sum = np.sum(predictions_df.iloc[:, narrowing_mask].to_numpy(), axis = 1)
    erosion_sum = np.sum(predictions_df.iloc[:, erosion_mask].to_numpy(), axis = 1)
    
    total_sum = narrowing_sum + erosion_sum
    
    predictions_df['Overall_narrowing'] = narrowing_sum
    predictions_df['Overall_erosion'] = erosion_sum
    predictions_df['Overall_Tol'] = total_sum
    
    template = pd.read_csv(config.test_template_path)
    predictions_df = predictions_df[template.columns]
    
    return predictions_df
    
def _get_test_datasets(config, hands_joint_source, feet_joints_source):
    test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = True, joint_scale = 5)

    hands_test_dataset, _ = test_dataset.get_hands_joint_test_dataset(joints_source = hands_joint_source)
    
    wrist_test_dataset = joint_test_dataset(config, config.train_fixed_location)
    wrists_test_dataset, _ = wrist_test_dataset.get_wrists_joint_test_dataset(joints_source = hands_joint_source)
    feet_test_dataset, _ = test_dataset.get_feet_joint_test_dataset(joints_source = feet_joints_source)

    return hands_test_dataset, wrists_test_dataset, feet_test_dataset

def _get_joint_damage_predictors(model_parameters_collection):
    hand_narrowing_predictor = joint_damage_predictor(model_parameters_collection['hands_narrowing_model'])
    wrists_narrowing_predictor = joint_damage_predictor(model_parameters_collection['wrists_narrowing_model'])
    feet_narrowing_predictor = joint_damage_predictor(model_parameters_collection['feet_narrowing_model'])

    hand_erosion_predictor = joint_damage_predictor(model_parameters_collection['hands_erosion_model'])
    wrists_erosion_predictor = joint_damage_predictor(model_parameters_collection['wrists_erosion_model'])
    feet_erosion_predictor = joint_damage_predictor(model_parameters_collection['feet_erosion_model'])

    return hand_narrowing_predictor, wrists_narrowing_predictor, feet_narrowing_predictor, hand_erosion_predictor, wrists_erosion_predictor, feet_erosion_predictor

def _get_details(file_info):
    file_info = file_info.numpy()
    
    img_info = file_info[0].decode('utf-8').split('.')[0]
    key = file_info[3].decode('utf-8')
    
    patient_info = img_info.split('-')
    patient_id = patient_info[0]
    part = patient_info[1]
        
    return patient_id, part, key