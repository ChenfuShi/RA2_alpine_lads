import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

from dataset import overall_joints_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor

test_models = {
    "hands_narrowing": "weights/SC1_v3A_hand_narrowing_maeafter_model_100.h5",
    "hands_erosion": "weights/SC1_v3A_hand_narrowing_maeafter_model_40.h5",
    "feet_narrowing": "weights/SC1_v3A_feet_narrowing_mae_sgd_cosineafter_model_100.h5",
    "feet_erosion": "weights/SC1_v3A_feet_erosion_mae_sgd_cosineafter_model_100.h5",
}

def _robust_mean(scores):
    scores = np.sort(scores)
    size_p = len(scores)//10
    mean_score = np.mean(scores[size_p:len(scores)-size_p])
    return min(max(0, mean_score),50)
    
def _prepare_image(img_arr):
    imgs = [np.expand_dims(x.numpy(),0) for x in img_arr]
    return imgs


def predict_overall(configuration, model_files = test_models, hand_joints_source = './data/predictions/hand_joint_data_test_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_test_v2.csv', patient_info = "/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/training_dataset/training.csv"):
    patient_df = pd.read_csv(patient_info)
    
    # hand narrowing
    hand_extractor = get_joint_extractor("H", False) 
    data_class = overall_joints_dataset.hands_overall_joints_dataset(configuration, 'test', joint_extractor = hand_extractor, force_augment = True)
    test_dataset = data_class.create_hands_overall_joints_dataset(patient_info,joints_source = hand_joints_source)
    patient_scoring = _predict_single(model_files["hands_narrowing"], test_dataset)
    patient_df["hands_narrowing"] = patient_scoring.sum(axis=1)
    # hand erosion
    hand_extractor = get_joint_extractor("H", True) 
    data_class = overall_joints_dataset.hands_overall_joints_dataset(configuration, 'test', joint_extractor = hand_extractor, force_augment = True)
    test_dataset = data_class.create_hands_overall_joints_dataset(patient_info,joints_source = hand_joints_source)
    patient_scoring = _predict_single(model_files["hands_erosion"], test_dataset)
    patient_df["hands_erosion"] = patient_scoring.sum(axis=1)
    # feet narrowing
    foot_extractor = get_joint_extractor("F", False) 
    data_class = overall_joints_dataset.feet_overall_joints_dataset(configuration, 'test', joint_extractor = foot_extractor, force_augment = True)
    test_dataset = data_class.create_feet_overall_joints_dataset(patient_info,joints_source = feet_joints_source)
    patient_scoring = _predict_single(model_files["feet_narrowing"], test_dataset)
    patient_df["feet_narrowing"] = patient_scoring.sum(axis=1)
    # feet erosion
    foot_extractor = get_joint_extractor("F", True) 
    data_class = overall_joints_dataset.feet_overall_joints_dataset(configuration, 'test', joint_extractor = foot_extractor, force_augment = True)
    test_dataset = data_class.create_feet_overall_joints_dataset(patient_info,joints_source = feet_joints_source)
    patient_scoring = _predict_single(model_files["feet_erosion"], test_dataset)
    patient_df["feet_erosion"] = patient_scoring.sum(axis=1)

    patient_df['Overall_narrowing'] = patient_df[["hands_narrowing","feet_narrowing"]].sum(axis=1)
    patient_df['Overall_erosion'] = patient_df[["hands_erosion","feet_erosion"]].sum(axis=1)
    patient_df['Overall_Tol'] = patient_df[["hands_narrowing","hands_erosion","feet_narrowing","feet_erosion"]].sum(axis=1)

    return patient_df


def _predict_single(model_file, test_dataset ):
    model = keras.models.load_model(model_file)
    results = {}

    for iteration in range(100):
        for x in test_dataset:
            imgs, img_id = _prepare_image(x[1]), x[0].numpy()[0].decode("utf-8")
            res = model.predict(imgs)
            try:
                results[img_id].append(res[0][0])
            except:
                results[img_id] = [res[0][0]]

    for item in results:
        results[item] = _robust_mean(results[item])

    result = pd.DataFrame.from_dict(results,orient='index')
    result.columns = ["Prediction"]
    result["patient"] = result.index.str.split("-").str[0]
    result["side"] = result.index.str.split("-").str[1]
    result = result.pivot(index = "patient", values="Prediction", columns="side")

    return result