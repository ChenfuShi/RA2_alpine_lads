########################################




########################################

from utils.config import Config
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


    # define model


    # train

