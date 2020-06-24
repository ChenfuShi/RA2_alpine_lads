import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

import os
from utils.config import Config
import model
import dataset
import logging
import dataset.NIH_pretrain_dataset as dpd
from model import NIH_model
from train.pretrain_NIH import pretrain_NIH_chest
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')
    
    # set up configuration
    configuration = Config()
    # after this logging.info() will print save to a specific file (global)!
    logging.info("configuration loaded")
    
    # prepare data
    logging.info("preparing train dataset")

    dataset = dpd.pretrain_dataset_NIH_chest(configuration)
    chest_dataset, chest_dataset_val = dataset.initialize_pipeline()

    logging.info("datasets prepared")
    # define model

    # no weights then go for new model
    model = NIH_model.create_small_dense(configuration, "small_wrist_dense_1M_NIH_nosex_256x256")
    model.summary()
    # check if there is weights to load
    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_NIH_chest(model, chest_dataset, chest_dataset_val, configuration, "small_wrist_dense_1M_NIH_nosex_256x256", epochs = 101)