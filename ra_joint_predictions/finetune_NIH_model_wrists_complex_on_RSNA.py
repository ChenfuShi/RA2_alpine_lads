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
wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, pad_resize = True).create_rsna_wrist_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_new_pretrain_model_0.h5", no_joint_types = 1, name = "RSNAonly_wrists")

model.summary()

finetune_model(model, 'complex_model_RSNA_wrists_pretrain_without_NIH_pretrain_fixed_joints_moreaug', wrist_dataset, wrist_val_dataset, epochs_before = 0, epochs_after = 101, n_outputs = 1)