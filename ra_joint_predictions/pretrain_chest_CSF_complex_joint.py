########################################




########################################

from utils.config import Config
import model
import dataset
import logging
import dataset.NIH_pretrain_dataset as dpd
from model import NIH_model
from train.pretrain_NIH import pretrain_NIH_chest
from tensorflow.keras.models import load_model

if __name__ == '__main__':

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
    model = NIH_model.create_complex_joint_multioutput(configuration)
    model.summary()
    # check if there is weights to load
    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,"NIH_new_pretrain")