########################################




########################################

from utils.config import Config
from dataset.dataset_preprocessing import fix_images
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

    logging.info("doing preprocessing")
    fix_images(configuration)
    logging.info("preprocessing done")
    dataset = train_dataset(configuration)
    hands_dataset,feet_dataset,hands_dataset_val,feet_dataset_val = dataset.initialize_pipeline()

    # define model
    

    # train

