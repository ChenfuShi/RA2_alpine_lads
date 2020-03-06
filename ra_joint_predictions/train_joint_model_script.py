import logging
import os
import sys

import tensorflow as tf

from utils.config import Config
from utils.saver import save_pretrained_model

from train.train_joints_damage import train_joints_damage_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')

    config = Config()
    logging.info('Command line arguments: ' + sys.argv)

    pretrained_model = sys.argv[1]
    logging.info('Using pretrainde model: %s', pretrained_model)

    model_name = sys.argv[2]
    logging.info('Saving trained model to: %s', model_name)

    # F for feet, H for hands, W for wrists
    joint_type = sys.argv[3]
    # E for Erosion, J for narrowing
    dmg_type = sys.argv[4]

    do_validation = sys.argv[5] == 'Y'

    # load pretrained model
    loaded_model = tf.keras.models.load_model('./pretrained_models/' + pretrained_model + '.h5')

    trained_model, hist_df = train_joints_damage_model(config, model_name, loaded_model, joint_type, dmg_type, do_validation = do_validation)

    save_pretrained_model(trained_model, 0, './pretrained_models/' + model_name)
    hist_df.to_csv('./pretrained_models/hist/' + model_name + '_hist.csv')
