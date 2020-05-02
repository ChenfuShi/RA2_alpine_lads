import logging
import numpy as np
import pandas as pd
import tensorflow as tf

import dataset.joint_dataset as joint_dataset

from dataset.test_dataset import joint_test_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor

from dataset.joints.joint_extractor import default_joint_extractor
from prediction.joint_damage_prediction import joint_damage_type_predictor, joint_damage_predictor, augmented_predictor, filtered_joint_damage_predictor

def predict_test_set(config, model_parameters_collection, hands_joint_source = './data/predictions/hand_joint_data_test_v2.csv', feet_joint_source = './data/predictions/feet_joint_data_test_v2.csv'):
    datasets = _get_test_datasets(config, hands_joint_source, feet_joint_source)
    hand_narrowing_predictor, wrists_narrowing_predictor, feet_narrowing_predictor, hand_erosion_predictor, wrists_erosion_predictor, feet_erosion_predictor = _get_joint_damage_predictors(model_parameters_collection)
    
    preds = {}
    
    def _init_preds(patient_id):
        if not patient_id in preds.keys():
            preds[patient_id] = {
                'Patient_ID': patient_id
        }

    def _predict(outcome_mapping, narrowing_predictor, narrowing_dataset, erosion_predictor, erosion_dataset):
        for file_info, img in narrowing_dataset:
            patient_id, part, key = _get_details(file_info)
            if len(outcome_mapping[key][0]) > 0:
                _init_preds(patient_id)

                y_preds = narrowing_predictor.predict_joint_damage(img)
                narrowing_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][0]]
                for idx, narrowing_outcome_key in enumerate(narrowing_outcome_keys):
                    y_pred = y_preds[idx]

                    logging.info(f"Predicted narrowing {y_pred} for {key} for patient {patient_id}_{narrowing_outcome_key}")

                    preds[patient_id][narrowing_outcome_key] = y_pred

        for file_info, img in erosion_dataset:
            patient_id, part, key = _get_details(file_info)
        
            if len(outcome_mapping[key][1]) > 0:
                _init_preds(patient_id)

                y_preds = erosion_predictor.predict_joint_damage(img)
                erosion_outcome_keys = [outcome_key.format(part = part) for outcome_key in outcome_mapping[key][1]]
                for idx, erosion_outcome_key in enumerate(erosion_outcome_keys):
                    y_pred = y_preds[idx]

                    logging.info(f"Predicted erosion {y_pred} for {key} for patient {patient_id}_{erosion_outcome_key}")

                    preds[patient_id][erosion_outcome_key] = y_pred

    logging.info("Predicting hands")
    _predict(joint_dataset.hand_outcome_mapping, hand_narrowing_predictor, datasets['hands_narrowing_dataset'], hand_erosion_predictor, datasets['hands_erosion_dataset'])
    logging.info("Predicting wrists")
    _predict(joint_dataset.wrist_outcome_mapping, wrists_narrowing_predictor, datasets['wrists_narrowing_dataset'], wrists_erosion_predictor, datasets['wrists_erosion_dataset'])
    logging.info("Predicting feet")
    _predict(joint_dataset.foot_outcome_mapping, feet_narrowing_predictor, datasets['feet_narrowing_dataset'], feet_erosion_predictor, datasets['feet_erosion_dataset'])

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
    df_joint_extractor = default_joint_extractor(joint_scale = 5)
    df_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = df_joint_extractor)
    
    hands_narrowing_extractor = get_joint_extractor('H', False)
    hands_narrowing_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = hands_narrowing_extractor)
    
    feet_narrowing_extractor = get_joint_extractor('F', False)
    feet_narrowing_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = feet_narrowing_extractor)

    hands_erosion_extractor = get_joint_extractor('H', True)
    hands_erosion_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = hands_erosion_extractor)
    
    feet_erosion_extractor = get_joint_extractor('F', True)
    feet_erosion_test_dataset = joint_test_dataset(config, config.train_fixed_location, pad_resize = False, joint_extractor = feet_erosion_extractor)
    
    return {
        'hands_narrowing_dataset': hands_narrowing_test_dataset.get_hands_joint_test_dataset(joints_source = hands_joint_source)[0],
        'wrists_narrowing_dataset': df_test_dataset.get_wrists_joint_test_dataset(joints_source = hands_joint_source)[0],
        'feet_narrowing_dataset': feet_narrowing_test_dataset.get_feet_joint_test_dataset(joints_source = feet_joints_source)[0],
        'hands_erosion_dataset': hands_erosion_test_dataset.get_hands_joint_test_dataset(joints_source = hands_joint_source)[0],
        'wrists_erosion_dataset': df_test_dataset.get_wrists_joint_test_dataset(joints_source = hands_joint_source)[0],
        'feet_erosion_dataset': feet_erosion_test_dataset.get_feet_joint_test_dataset(joints_source = feet_joints_source)[0]
    }

def _get_joint_damage_predictors(model_parameters_collection):
    def _get_predictor(model_parameters, filter = True):
        dmg_pred = augmented_predictor(joint_damage_predictor(model_parameters), no_augments = 50, rounding_cutoff = 0.3)

        if filter is True:
            filter_pred = augmented_predictor(joint_damage_type_predictor(model_parameters), no_augments = 50)

            dmg_pred = filtered_joint_damage_predictor(model_parameters, filter_pred, dmg_pred)

        return dmg_pred

    hand_narrowing_predictor = _get_predictor(model_parameters_collection['hands_narrowing_model'])
    wrists_narrowing_predictor = _get_predictor(model_parameters_collection['wrists_narrowing_model'], filter = False)
    feet_narrowing_predictor = _get_predictor(model_parameters_collection['feet_narrowing_model'])

    hand_erosion_predictor = _get_predictor(model_parameters_collection['hands_erosion_model'])
    wrists_erosion_predictor = _get_predictor(model_parameters_collection['wrists_erosion_model'], filter = False) 
    feet_erosion_predictor = _get_predictor(model_parameters_collection['feet_erosion_model'])

    return hand_narrowing_predictor, wrists_narrowing_predictor, feet_narrowing_predictor, hand_erosion_predictor, wrists_erosion_predictor, feet_erosion_predictor

def _get_details(file_info):
    file_info = file_info.numpy()
    
    img_info = file_info[0].decode('utf-8').split('.')[0]
    key = file_info[3].decode('utf-8')
    
    patient_info = img_info.split('-')
    patient_id = patient_info[0]
    part = patient_info[1]
        
    return patient_id, part, key
