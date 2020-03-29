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
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(20)

import dataset.combined_joints_dataset as combset
from prediction import joint_damage
from model import combined_sc1_model
from train import combined_sc1_train

# need this to stop that warning
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train_feet():
    
    combined_model_feet = combined_sc1_model.get_feet_model("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/feet_narrowing_reg_old_pretrain_adam_shuffle.h5")

    train_dataset_class = combset.overall_test_feet(configuration,"/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/training_dataset/fixed")
    train_data = train_dataset_class.create_generator(outcomes_source = "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv")

    val_dataset_class = combset.overall_test_feet(configuration,"/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/training_dataset/fixed")
    val_data = train_dataset_class.create_generator(outcomes_source = "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv",joints_source= "./data/predictions/feet_joint_data_test_v2.csv")

    combined_sc1_train.train_SC1_model(configuration,combined_model_feet,"SC1_v3A_feet_narrowing_0.4zeros",train_data,val_data,epochs_before = 10, epochs_after = 51)


def train_hands():
        
    combined_model_hands = combined_sc1_model.get_hand_model("/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/hands_narrowing_reg_old_pretrain_adam_shuffle.h5",
                                                            "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/trained_models/v3/wrists_narrowing_reg_old_pretrain_adam_shuffle.h5")


    train_dataset_class = combset.overall_test_hand(configuration,"/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/training_dataset/fixed")
    train_data = train_dataset_class.create_generator(outcomes_source = "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv")

    val_dataset_class = combset.overall_test_hand(configuration,"/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/training_dataset/fixed")
    val_data = train_dataset_class.create_generator(outcomes_source = "/mnt/iusers01/jw01/mdefscs4/ra_challenge/RA_challenge/training_dataset/training.csv",joints_source= "./data/predictions/hand_joint_data_test_v2.csv")

    combined_sc1_train.train_SC1_model(configuration,combined_model_hands,"SC1_v3A_hand_narrowing_0.4zeros",train_data,val_data,epochs_before = 10, epochs_after = 51)


# lose the variables because it fucks memory or something
train_feet()
train_hands()