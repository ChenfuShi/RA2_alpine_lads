from utils.config import Config
import model
import dataset
import logging

from model import RSNA_model
from train.pretrain_RSNA_joints import finetune_model
from tensorflow.keras.models import load_model


configuration = Config()

model = RSNA_model.complex_joint_finetune_model(configuration,weights="weights/NIH_new_pretrain_model_75")

model.summary()

finetune_model(model,"complex_model_RSNA_pretrain_with_NIH_pretrain",configuration)