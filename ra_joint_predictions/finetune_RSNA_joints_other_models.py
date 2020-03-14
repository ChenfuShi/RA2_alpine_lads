from utils.config import Config
import model
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset


configuration = Config()

## joints

joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration).create_rsna_joints_dataset(val_split = True)


for model_file, name in zip([],[]):

    model = RSNA_model.complex_joint_finetune_model(configuration,weights=model_file)

    model.summary()

    finetune_model(model,"{name}_RSNA_pretrain_with_NIH_pretrain",joint_dataset,joint_val_dataset, epochs_before = 11, epochs_after = 101)
