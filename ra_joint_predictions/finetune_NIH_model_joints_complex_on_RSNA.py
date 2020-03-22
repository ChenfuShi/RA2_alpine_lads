from utils.config import Config
import model
import os
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset

os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

configuration = Config()

## joints
joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, pad_resize = True, joint_scale = 5).create_rsna_joints_dataset(val_split = True, include_wrist_joints = True)

model = RSNA_model.complex_joint_finetune_model(configuration, weights = "weights/NIH_new_pretrain_model_0.h5", no_joint_types = 13)

model.summary()

finetune_model(model, "complex_model_RSNA_joints_pretrain_without_NIH_pretrain_fixed_joints_moreaug", joint_dataset, joint_val_dataset, n_outputs = 13, epochs_before = 0, epochs_after = 101)
