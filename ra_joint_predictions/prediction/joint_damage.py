import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from dataset.test_dataset import joint_test_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor

from dataset.joints.joint_extractor import default_joint_extractor
from prediction.joint_damage_prediction import ensembled_filter_predictor

def predict_test_set(config, model_parameters_collection, hands_joint_source = './data/predictions/hand_joint_data_test_v2.csv', feet_joint_source = './data/predictions/feet_joint_data_test_v2.csv', hands_invalid_images = None, feet_invalid_images = None):
    datasets = _get_test_datasets(config, hands_joint_source, feet_joint_source)
    
    preds = {}
    
    def _init_preds(patient_id):
        if not patient_id in preds.keys():
            preds[patient_id] = {
                'Patient_ID': patient_id
            }
            
    def _predict(joint_type, outcome_mapping):
        dataset = datasets[f'{joint_type}_dataset']
        
        narrowing_predictor = _get_predictor(model_parameters_collection[f'{joint_type}_narrowing_model'])
        erosion_predictor = _get_predictor(model_parameters_collection[f'{joint_type}_erosion_model'])
        
        for file_info, img in dataset:
            patient_id, part, key = _get_details(file_info)
           
            if len(outcome_mapping[key][0]) > 0:
                _init_preds(patient_id)

                y_preds, pred_labels = narrowing_predictor.predict_joint_damage(img)
                narrowing_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][0]]
                for idx, narrowing_outcome_key in enumerate(narrowing_outcome_keys):
                    y_pred = y_preds[idx]
                    pred_label = pred_labels[idx]
                    
                    logging.info(f"Predicted narrowing {y_pred} for patient {patient_id}_{narrowing_outcome_key} ({pred_label})")

                    preds[patient_id][narrowing_outcome_key] = y_pred
            
            if len(outcome_mapping[key][1]) > 0:
                _init_preds(patient_id)

                y_preds, pred_labels = erosion_predictor.predict_joint_damage(img)
                erosion_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][1]]
                for idx, erosion_outcome_key in enumerate(erosion_outcome_keys):
                    y_pred = y_preds[idx]
                    pred_label = pred_labels[idx]

                    logging.info(f"Predicted erosion {y_pred} for patient {patient_id}_{erosion_outcome_key} ({pred_label})")

                    preds[patient_id][erosion_outcome_key] = y_pred
                    
        logging.info('-------------')        
        
        n_filtered = narrowing_predictor.n_filtered_images
        n_images = narrowing_predictor.n_processed_images
            
        filtered_ratio = n_filtered / n_images
            
        logging.info(f'{joint_type} narrowing filtered {filtered_ratio} of joints ({n_filtered} / {n_images})')
            
        
        n_filtered = erosion_predictor.n_filtered_images
        n_images = erosion_predictor.n_processed_images
            
        filtered_ratio = n_filtered / n_images
            
        logging.info(f'{joint_type} erosion filtered {filtered_ratio} of joints ({n_filtered} / {n_images})')
        logging.info('-------------') 
        logging.info('-------------')    

    logging.info("Predicting hands")
    _predict('hands', joint_dataset.hand_outcome_mapping)
    logging.info("Predicting wrists")
    _predict('wrists', joint_dataset.wrist_outcome_mapping)
    logging.info("Predicting feet")
    _predict('feet', joint_dataset.foot_outcome_mapping)
    
    logging.info("Adding Missing Predictions")

    missing_preds = _get_missing_preds(train_df, hands_invalid_images, feet_invalid_images)
    for patient_id in missing_preds:
        _init_preds(patient_id)
        
        for outcome_key in missing_preds[patient_id]['missing_values']:
            logging.info(f'Adding missing outcome {outcome_key} for patient {patient_id}')
            
            preds[patient_id][outcome_key] = missing_preds[patient_id]['missing_values'][outcome_key]

    predictions_df = pd.DataFrame(preds.values(), index = np.arange(len(preds.values())))
    
    logging.info("Calculate damage totals")

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
    
    logging.info("Finished predictions")

    return predictions_df
    
def _get_test_datasets(config, hands_joint_source, feet_joints_source):
    hands_extractor = get_joint_extractor('H', False)
    hands_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = hands_extractor)
    
    df_joint_extractor = default_joint_extractor(joint_scale = 5)
    wrists_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = df_joint_extractor)
    
    feet_extractor = get_joint_extractor('F', False)
    feet_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = feet_extractor)
    
    return {
        'hands_dataset': hands_test_dataset.get_hands_joint_test_dataset(joints_source = hands_joint_source)[0],
        'wrists_dataset': wrists_test_dataset.get_wrists_joint_test_dataset(joints_source = hands_joint_source)[0],
        'feet_dataset': feet_test_dataset.get_feet_joint_test_dataset(joints_source = feet_joints_source)[0]
    }

def _get_predictor(model_parameters):
    return ensembled_filter_predictor(model_parameters, no_augments = 50, rounding_cutoff = 0.3)

def _get_details(file_info):
    file_info = file_info.numpy()
    
    img_info = file_info[0].decode('utf-8').split('.')[0]
    key = file_info[3].decode('utf-8')
    
    patient_info = img_info.split('-')
    patient_id = patient_info[0]
    part = patient_info[1]
        
    return patient_id, part, key

def _get_missing_preds(train_df, hands_invalid_images, feet_invalid_images):
    mean_train_df = train_df.mean()
    
    missing_preds = {}
    
    def _init_missing_preds(patient_id):
        if not patient_id in missing_preds.keys():
            missing_preds[patient_id] = {
                'missing_values': {}
            }
    
    def _add_missing_preds_for_outcome_mapping(patient_id, outcome_mapping):
        for key in outcome_mapping:
            for outcome_key in outcome_mapping[key][0]:
                outcome_key = outcome_key.format(part = part)
                
                missing_preds[patient_id]['missing_values'][outcome_key] = mean_train_df[outcome_key]
                
            for outcome_key in outcome_mapping[key][1]:
                outcome_key = outcome_key.format(part = part)
                
                missing_preds[patient_id]['missing_values'][outcome_key] = mean_train_df[outcome_key]
    
    if hands_invalid_images is not None:
        for hand_invalid_image in hands_invalid_images:
            details = hand_invalid_image.split('-')

            patient_id = details[0]
            part = details[1]

            _init_missing_preds(patient_id)

            _add_missing_preds_for_outcome_mapping(patient_id, joint_dataset.hand_outcome_mapping)
            _add_missing_preds_for_outcome_mapping(patient_id, joint_dataset.wrist_outcome_mapping)
        
    if feet_invalid_images is not None:
        for feet_invalid_image in feet_invalid_images:
            details = feet_invalid_image.split('-')

            patient_id = details[0]
            part = details[1]

            _init_missing_preds(patient_id)

            _add_missing_preds_for_outcome_mapping(patient_id, joint_dataset.foot_outcome_mapping)
                
    return missing_preds