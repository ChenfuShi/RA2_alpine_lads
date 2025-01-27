import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from utils.config import Config
import PIL
import json
import logging

configuration = Config()
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)
from dataset.joints.joint_extractor_factory import get_joint_extractor
from dataset import overall_joints_dataset
from prediction import joint_damage
from model import combined_sc1_model
from train import combined_sc1_train

def train_feet():
    feet_extractor = get_joint_extractor("F", False)
    combined_model_feet = combined_sc1_model.get_feet_model("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/feet_narrowing_reg_old_pretrain_adam_shuffle.h5")

    ds, val_ds, no_val_samples = overall_joints_dataset.feet_overall_joints_dataset(configuration, 'train', joint_extractor = feet_extractor, erosion_flag = False).create_feet_overall_joints_dataset_with_validation("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv")

    combined_sc1_train.train_SC1_model(configuration,combined_model_feet,"SC1_v3A_feet_narrowing_mae_adamw3",ds,val_ds,epochs_before = 11, epochs_after = 101)
    
    feet_extractor = get_joint_extractor("F", True)
    combined_model_feet = combined_sc1_model.get_feet_model("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/feet_erosion_reg_old_pretrain_adam_shuffle.h5")

    ds, val_ds, no_val_samples = overall_joints_dataset.feet_overall_joints_dataset(configuration, 'train', joint_extractor = feet_extractor, erosion_flag = True).create_feet_overall_joints_dataset_with_validation("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv")

    combined_sc1_train.train_SC1_model(configuration,combined_model_feet,"SC1_v3A_feet_erosion_mae_adamw3",ds,val_ds,epochs_before = 11, epochs_after = 101)


def train_hands():
    hand_extractor = get_joint_extractor("H", False)
    combined_model_hands = combined_sc1_model.get_hand_model("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/hands_narrowing_reg_old_pretrain_adam_shuffle.h5",
                                                            "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/wrists_narrowing_reg_old_pretrain_adam_shuffle.h5")

    ds, val_ds, no_val_samples = overall_joints_dataset.hands_overall_joints_dataset(configuration, 'train', joint_extractor = hand_extractor, erosion_flag = False).create_hands_overall_joints_dataset_with_validation("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv")

    combined_sc1_train.train_SC1_model(configuration,combined_model_hands,"SC1_v3A_hand_narrowing_mae_adamw3",ds,val_ds,epochs_before = 11, epochs_after = 101)
    
    hand_extractor = get_joint_extractor("H", True)
    combined_model_hands = combined_sc1_model.get_hand_model("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/hands_erosion_reg_old_pretrain_adam_shuffle.h5",
                                                            "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/wrists_erosion_reg_old_pretrain_adam_shuffle.h5",
                                                            erosion_flag = True)

    ds, val_ds, no_val_samples = overall_joints_dataset.hands_overall_joints_dataset(configuration, 'train', joint_extractor = hand_extractor, erosion_flag = True).create_hands_overall_joints_dataset_with_validation("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv")

    combined_sc1_train.train_SC1_model(configuration,combined_model_hands,"SC1_v3A_hand_erosion_mae_adamw3",ds,val_ds,epochs_before = 11, epochs_after = 101)


# lose the variables because it fucks memory or something


train_feet()
train_hands()