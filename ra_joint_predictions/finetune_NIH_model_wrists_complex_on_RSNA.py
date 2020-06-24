import tensorflow as tf

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

model = RSNA_model.finetune_rsna_model(configuration, './weights/small_bottlenecked_vgg_550k_NIH_nosex_256x256_model_100.h5', 'small_bottlenecked_vgg_550k_wrist_nosex_192x256_adamW_200', no_joint_types = 1)

# model = RSNA_model.create_small_densenet(configuration, 'small_2M_dense_wrist_bottleneck_gap_nonfc_renorm_nosex_adamW_200', no_joint_types = 1)

model.summary()

finetune_model(model, 'small_bottlenecked_vgg_550k_wrist_nosex_192x256_adamW_200', wrist_dataset, wrist_val_dataset, epochs_before = 25, epochs_after = 201, n_outputs = 1, is_wrists = True)
