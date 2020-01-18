########################################




########################################

from utils.config import Config
from dataset.train_dataset import train_dataset
import model
import dataset
import logging

class CSF_test_config(Config):
    """
    Overwrite configuration
    """
    pass


if __name__ == '__main__':

    # set up configuration
    configuration = CSF_test_config()
    # after this logging.info() will print save to a specific file (global)!
    logging.info("configuration loaded")
    
    # prepare data
    logging.info("preparing train dataset")
    dataset = train_dataset(configuration)
    hands_dataset,feet_dataset,hands_dataset_val,feet_dataset_val = dataset.initialize_pipeline()

    # define model


    # train

