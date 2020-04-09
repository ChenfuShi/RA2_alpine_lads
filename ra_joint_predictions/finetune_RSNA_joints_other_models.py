from utils.config import Config
import model
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model
from dataset.rsna_joint_dataset import rsna_joint_dataset, rsna_wrist_dataset
from dataset.joints.joint_extractor_factory import get_joint_extractor

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)



configuration = Config()
# rsna_extractor = get_joint_extractor("RSNA", True)
# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, pad_resize = False, joint_extractor = rsna_extractor).create_rsna_joints_dataset(val_split = True)

# "NIH_new_pretrain_model_100.h5", "NIH_densenet_model_100.h5", "NIH_Xception_model_100.h5",
# "complex_with_NIH_nopad_erosion_shape", "densenet_with_NIH_nopad_erosion_shape","Xception_with_NIH_nopad_erosion_shape",

# models = ["NIH_rewritten_model_100.h5"]

# names = ["rewritten_with_NIH_nopad_erosion_shape"]

# for model_file, name in zip(models,names):

#     model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/" + model_file, no_joint_types = 13, name = name + "_RSNA_joint")

#     model.summary()

#     finetune_model(model,f"{name}_RSNA_joint_v4",joint_dataset,joint_val_dataset, epochs_before = 10, epochs_after = 76, n_outputs = 13)


# "NIH_new_pretrain_model_0.h5",  "NIH_new_pretrain_model_100.h5", "NIH_densenet_model_100.h5", "NIH_Xception_model_100.h5", 
# "complex_no_NIH_nopad","complex_with_NIH_nopad_new_shape", "densenet_with_NIH_nopad_new_shape","Xception_with_NIH_nopad_new_shape", 

models = ["NIH_rewritten_model_0.h5"]

names = ["rewritten_no_NIH_nopad_new_shape"]

wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

for model_file, name in zip(models,names):

    model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/" + model_file, no_joint_types = 1, name = name + "_RSNA_wrist")

    model.summary()

    finetune_model(model,f"{name}_RSNA_wrist_v3",wrist_dataset,wrist_val_dataset, epochs_before = 10, epochs_after = 51, n_outputs = 1)



# imagenet models

# configuration = Config()
# name = "densent_imagenet_with_NIH_nopad_erosion_shape"
# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, imagenet = True, pad_resize = False, joint_extractor = rsna_extractor).create_rsna_joints_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_densenet_imagenet_model_0.h5", no_joint_types = 13, name = name + "_RSNA_joint")

# model.summary()

# finetune_model(model,f"{name}_RSNA_joint_v4",joint_dataset,joint_val_dataset, epochs_before = 10, epochs_after = 76, n_outputs = 13)


# wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, imagenet = True, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_densenet_imagenet_model_100.h5", no_joint_types = 1, name = name + "_RSNA_wrist")

# model.summary()

# finetune_model(model,f"{name}_RSNA_wrist_v3",wrist_dataset,wrist_val_dataset, epochs_before = 10, epochs_after = 51, n_outputs = 1)



# configuration = Config()
# name = "Xception_imagenet_with_NIH_nopad_erosion_shape"
# configuration.img_height = 299
# configuration.img_width = 299
# configuration.joint_img_height = 299
# configuration.joint_img_width = 299
# configuration.batch_size = 32

# joint_dataset, joint_val_dataset = rsna_joint_dataset(configuration, imagenet = True, pad_resize = False, joint_extractor = rsna_extractor).create_rsna_joints_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_Xception_imagenet_model_100.h5", no_joint_types = 13, name = name + "_RSNA_joint")

# model.summary()

# finetune_model(model,f"{name}_RSNA_joint_v4",joint_dataset,joint_val_dataset, epochs_before = 10, epochs_after = 76, n_outputs = 13)


# wrist_dataset, wrist_val_dataset = rsna_wrist_dataset(configuration, imagenet = True, pad_resize = False).create_rsna_wrist_dataset(val_split = True)

# model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_Xception_imagenet_model_100.h5", no_joint_types = 1, name = name + "_RSNA_wrist")

# model.summary()

# finetune_model(model,f"{name}_RSNA_wrist_v3",wrist_dataset,wrist_val_dataset, epochs_before = 10, epochs_after = 51, n_outputs = 1)