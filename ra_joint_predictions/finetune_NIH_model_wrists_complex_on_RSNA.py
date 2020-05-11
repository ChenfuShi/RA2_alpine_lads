import tensorflow as tf
i

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

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
wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

model = RSNA_model.finetune_rsna_model(configuration, "weights/NIH_complex_gap_adam_model_100.h5", no_joint_types = 1, name = "RSNA_gap_wrists")

model.summary()

finetune_model(model, 'complex_gap_model_RSNA_wrists_pretrain', wrist_dataset, wrist_val_dataset, epochs_before = 0, epochs_after = 51, n_outputs = 1, is_wrists = True)
