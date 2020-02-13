import logging
import os
import sys

from train.pretrain import pretrain_rnsa_multioutput_model
from model.rsna_multioutput_model import create_rsna_NASnet_multioutupt

from utils.config import Config
from utils import save_pretrained_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/RA2_alpine_lads/')

    logging.info('Loading config')
    config = Config()

    logging.info('Starting training')
    trained_model = pretrain_rnsa_multioutput_model('rsna_multioutput_NASnet_pretrain', config, create_rsna_NASnet_multioutupt)

    model_name = sys.argv[1]
    save_pretrained_model(trained_model, 3, model_name)