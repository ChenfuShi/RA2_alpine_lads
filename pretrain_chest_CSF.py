########################################




########################################

from utils.config import Config
from dataset.dataset_preprocessing import fix_images
import model
import dataset
import logging
import dataset.pretrain_dataset as dpd
from model import NASnet_multioutput
from utils.pretrain import pretrain_NIH_chest
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
    model = NASnet_multioutput.create_NASnet_multioutupt(configuration)
    # check if there is weights to load
    #model.load_weights("weights/NIH_chest_NASnet_model_325")
    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,"run_2_clean_mobileNASnet")