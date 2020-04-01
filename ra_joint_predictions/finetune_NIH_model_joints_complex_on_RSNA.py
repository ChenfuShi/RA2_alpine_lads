import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

from utils.config import Config
import model
import os
import dataset
import logging

import dataset.joint_dataset as joint_dataset

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset
from dataset.joints.joint_extractor import width_based_joint_extractor

os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

configuration = Config()

joint_extractor = width_based_joint_extractor(joint_scale = 4.5, height_scale = 1.2)

## joints
joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, pad_resize = False, joint_extractor = joint_extractor).create_rsna_joints_dataset(val_split = True, include_wrist_joints = True)

model = RSNA_model.complex_joint_finetune_model(configuration, weights = "../../../RA2_alpine_lads/ra_joint_predictions/weights/NIH_rewritten_model_50.h5", no_joint_types = 13)

model.summary()

finetune_model(model, "complex_rewritten_pretrain_full_adam_erosion_joints", joint_dataset, joint_val_dataset, n_outputs = 13, epochs_before = 25, epochs_after = 201)
