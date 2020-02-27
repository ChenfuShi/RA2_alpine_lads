import logging
import os
import sys

from train.pretrain_RSNA_joints import pretrain_rnsa_multioutput_model
from model.RSNA_model import create_rsna_NASnet_multioutupt

from utils.config import Config
from utils.saver import save_pretrained_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    model_name = sys.argv[1]
    logging.info('Saving output model to:', model_name)

    logging.info('Loading config')
    config = Config()

    logging.info('Starting training')
    trained_model = pretrain_rnsa_multioutput_model('rsna_multioutput_NASnet_pretrain', config, create_rsna_NASnet_multioutupt)

    save_pretrained_model(trained_model, 6, model_name)
