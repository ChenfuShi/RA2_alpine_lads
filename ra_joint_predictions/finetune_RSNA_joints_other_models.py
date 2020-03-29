from utils.config import Config
import model
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset


configuration = Config()

# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, pad_resize = False, joint_scale = 3.5).create_rsna_joints_dataset(val_split = True)



# models = ["NIH_new_pretrain_model_0.h5", "NIH_densenet_model_0.h5", "NIH_Xception_model_0.h5"]

# names = ["complex_no_NIH_scale3.5","densenet_no_NIH_scale3.5","Xception_no_NIH_scale3.5"]

# for model_file, name in zip(models,names):

#     model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/" + model_file, no_joint_types = 13, name = name + "_RSNA_joint")

#     model.summary()

#     finetune_model(model,f"{name}_RSNA_joint_v2",joint_dataset,joint_val_dataset, epochs_before = 0, epochs_after = 76, n_outputs = 13)


# "NIH_new_pretrain_model_0.h5", 
# "complex_no_NIH_nopad",

models = [ "NIH_densenet_model_0.h5", "NIH_Xception_model_0.h5", "NIH_new_pretrain_model_0.h5", ]

names = ["densenet_no_NIH_nopad","Xception_no_NIH_nopad", "complex_no_NIH_nopad",]

wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

for model_file, name in zip(models,names):

    model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/" + model_file, no_joint_types = 1, name = name + "_RSNA_wrist")

    model.summary()

    finetune_model(model,f"{name}_RSNA_wrist_v2",wrist_dataset,wrist_val_dataset, epochs_before = 0, epochs_after = 51, n_outputs = 1)



# imagenet models

# configuration = Config()
name = "densent_imagenet_no_NIH_nopad"
# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, imagenet = True, pad_resize = False, joint_scale = 3.5).create_rsna_joints_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_densenet_imagenet_model_0.h5", no_joint_types = 13, name = name + "_RSNA_joint")

# model.summary()

# finetune_model(model,f"{name}_RSNA_joint_v2",joint_dataset,joint_val_dataset, epochs_before = 0, epochs_after = 76, n_outputs = 13)


wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, imagenet = True, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_densenet_imagenet_model_0.h5", no_joint_types = 1, name = name + "_RSNA_wrist")

model.summary()

finetune_model(model,f"{name}_RSNA_wrist_v2",wrist_dataset,wrist_val_dataset, epochs_before = 0, epochs_after = 51, n_outputs = 1)



# configuration = Config()
# name = "Xception_imagenet_no_NIH_scale3.5"
# configuration.img_height = 299
# configuration.img_width = 299
# configuration.joint_img_height = 299
# configuration.joint_img_width = 299
# configuration.batch_size = 32

# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, imagenet = True, pad_resize = False, joint_scale = 3.5).create_rsna_joints_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_Xception_imagenet_model_0.h5", no_joint_types = 13, name = name + "_RSNA_joint")

# model.summary()

# finetune_model(model,f"{name}_RSNA_joint_v2",joint_dataset,joint_val_dataset, epochs_before = 0, epochs_after = 76, n_outputs = 13)


# wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, imagenet = True, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_Xception_imagenet_model_0.h5", no_joint_types = 1, name = name + "_RSNA_wrist")

# model.summary()

# finetune_model(model,f"{name}_RSNA_wrist_v2",wrist_dataset,wrist_val_dataset, epochs_before = 11, epochs_after = 51, n_outputs = 1)