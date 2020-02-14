import logging
import os
import sys

import tensorflow as tf

from train.pretrain import pretrain_rnsa_multioutput_model
from model.rsna_multioutput_model import create_rsna_NASnet_multioutupt

from utils.config import Config
from utils import save_pretrained_model

from train.train import train_feet_erosion_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/')

    pretrained_model = sys.argv[1]
    logging.info('Using pretrainde model: %s', pretrained_model)

    model_name = sys.argv[2]
    logging.info('Saving trained model to: %s', model_name)

    logging.info('Loading config')
    config = Config()

    # load pretrained model
    loaded_model = tf.keras.models.load_model(pretrained_model + '.h5')

    trained_model = train_feet_erosion_model(config, loaded_model)

    save_pretrained_model(trained_model, 0, model_name)