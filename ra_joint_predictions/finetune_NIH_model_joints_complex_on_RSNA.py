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
from dataset.joints.joint_extractor_factory import get_joint_extractor

from model.utils.building_blocks_joints import get_joint_model_input, vvg_joint_model

os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

configuration = Config()

joint_extractor = get_joint_extractor('RSNA', False)

## joints
joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, pad_resize = False, joint_extractor = joint_extractor).create_rsna_joints_dataset(val_split = True, include_wrist_joints = False)

# model = RSNA_model.create_small_bottlenecked_vgg(configuration, 'small_bottlenecked_vgg_500k_gap_renorm_nohead_nosex_92x92_diffextractor_adamW_200', no_joint_types = 10)

model = RSNA_model.finetune_rsna_model(configuration, './weights/small_wrist_dense_1M_NIH_nosex_256x256_model_100.h5', 'small_bottlenecked_dense_1M_nosex_joints_224x224_adamW_200_withNIH', no_joint_types = 10)

#model = RSNA_model.finetune_rsna_model(configuration, './weights/small_bottlenecked_dense_1M_gap_renorm_nohead_nosex_92x92_diffextractor_now_adamW_200after_model_200.h5', 'small_bottlenecked_dense_1M_gap_renorm_nohead_nosex_92x92_diffextractor_now_adamW_200_withNIH', no_joint_types = 10)
model.summary()

finetune_model(model, "small_bottlenecked_dense_1M_nosex_joints_224x224_adamW_200_withNIH", joint_dataset, joint_val_dataset, n_outputs = 10, epochs_before = 25, epochs_after = 201)