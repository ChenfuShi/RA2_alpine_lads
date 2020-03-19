from utils.config import Config
import model
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset


configuration = Config()

joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration).create_rsna_joints_dataset(val_split = True)
# "NIH_new_pretrain_model_100.h5", "NIH_densenet_model_100.h5",
# "complex","densenet",

# models = [ "NIH_Xception_model_100.h5"]

# names = ["Xception"]

# for model_file, name in zip(models,names):

#     model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/" + model_file, no_joint_types = 13, name = name + "_RSNA_joint")

#     model.summary()

#     finetune_model(model,f"{name}_RSNA_joint_v2",joint_dataset,joint_val_dataset, epochs_before = 11, epochs_after = 101, n_outputs = 13)


# 
# 
models = ["NIH_new_pretrain_model_100.h5", "NIH_densenet_model_100.h5", "NIH_Xception_model_100.h5"]

names = ["complex","densenet","Xception"]

wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration).create_rsna_wrist_dataset(val_split = True)

for model_file, name in zip(models,names):

    model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/" + model_file, no_joint_types = 1, name = name + "_RSNA_joint")

    model.summary()

    finetune_model(model,f"{name}_RSNA_wrist_v2",wrist_dataset,wrist_val_dataset, epochs_before = 11, epochs_after = 101, n_outputs = 1)



# imagenet models

configuration = Config()
name = "densent_imagenet"
joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, imagenet = True).create_rsna_joints_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_densenet_imagenet_model_100.h5" + model_file, no_joint_types = 13, name = name + "_RSNA_joint")

model.summary()

finetune_model(model,"f{name}_RSNA_joint_v2",joint_dataset,joint_val_dataset, epochs_before = 11, epochs_after = 101, n_outputs = 13)


wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, imagenet = True).create_rsna_wrist_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_densenet_imagenet_model_100.h5" + model_file, no_joint_types = 1, name = name + "_RSNA_joint")

model.summary()

finetune_model(model,"f{name}_RSNA_wrist_v2",wrist_dataset,wrist_val_dataset, epochs_before = 11, epochs_after = 101, n_outputs = 1)



configuration = Config()
name = "Xception_imagenet"
configuration.img_height = 299
configuration.img_width = 299
configuration.batch_size = 32

joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, imagenet = True).create_rsna_joints_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_Xception_imagenet_model_100.h5" + model_file, no_joint_types = 13)

model.summary()

finetune_model(model,"{name}_RSNA_joint_v2",joint_dataset,joint_val_dataset, epochs_before = 11, epochs_after = 101, n_outputs = 13)


wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, imagenet = True).create_rsna_wrist_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_Xception_imagenet_model_100.h5" + model_file, no_joint_types = 1)

model.summary()

finetune_model(model,"{name}_RSNA_wrist_v2",wrist_dataset,wrist_val_dataset, epochs_before = 11, epochs_after = 101, n_outputs = 1)