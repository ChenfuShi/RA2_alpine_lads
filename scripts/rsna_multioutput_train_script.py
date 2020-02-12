import logging

from train.pretrain import pretrain_rnsa_multioutput_model
from model.rsna_multioutput_model import create_rsna_NASnet_multioutupt

from utils.config import Config

if __name__ == '__main__':
    logging.info('Loading config')
    config = Config()

    logging.info('Starting training')
    pretrain_rnsa_multioutput_model('rsna_multioutput_NASnet_pretrain', config, create_rsna_NASnet_multioutupt)