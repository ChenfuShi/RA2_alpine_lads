import logging
import os
import sys

import tensorflow as tf

from utils.config import Config
from utils.saver import save_pretrained_model

from train.train_joints_damage import train_joints_damage_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    pretrained_model = sys.argv[1]
    logging.info('Using pretrainde model: %s', pretrained_model)

    model_name = sys.argv[2]
    logging.info('Saving trained model to: %s', model_name)

    # F for feet, H for hands, W for wrists
    joint_type = sys.argv[3]
    # E for Erosion, J for narrowing
    dmg_type = sys.argv[4]

    logging.info('Loading config')
    config = Config()

    # load pretrained model
    loaded_model = tf.keras.models.load_model(pretrained_model + '.h5')

    trained_model = train_joints_damage_model(config, model_name, loaded_model, joint_type, dmg_type)

    save_pretrained_model(trained_model, 0, model_name)
