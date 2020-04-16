from utils.config import Config
import model
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor

from tensorflow import keras 
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)



configuration = Config()

rsna_extractor = get_joint_extractor("RSNA", True)
joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, pad_resize = False, joint_extractor = rsna_extractor).create_rsna_joints_dataset(val_split = True)

models = ["NIH_rewritten_elu_a0.1_model_100.h5"]

names = ["rewritten_elu_a0.1_with_NIH_nopad_erosion_shape"]

for model_file, name in zip(models,names):

    model = RSNA_model.model_finetune_RSNA(configuration,weights="weights/" + model_file, no_joint_types = 13, name = name + "_RSNA_joint",act=(lambda x: keras.activations.elu(x, alpha = 0.1)))

    model.summary()

    finetune_model(model,f"{name}_RSNA_joint_v4",joint_dataset,joint_val_dataset, epochs_before = 10, epochs_after = 76, n_outputs = 13)
