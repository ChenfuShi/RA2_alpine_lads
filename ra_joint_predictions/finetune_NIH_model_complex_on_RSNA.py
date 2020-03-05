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

# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration).create_rsna_joints_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_new_pretrain_model_100.h5")

# model.summary()

# finetune_model(model,"complex_model_RSNA_pretrain_with_NIH_pretrain",joint_dataset,joint_val_dataset)



## wrists

wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration).create_rsna_wrist_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,1,weights="weights/NIH_new_pretrain_model_100.h5")

model.summary()

finetune_model(model,"complex_model_RSNA_pretrain_with_NIH_pretrain",wrist_dataset,wrist_val_dataset, n_outputs = 1)
