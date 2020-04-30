import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

import logging
import os
os.environ['TF_KERAS'] = '1'
import sys

from utils.config import Config
from utils.saver import save_pretrained_model

from train.train_joints_damage import train_joints_damage_model

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/michael_dev/RA2_alpine_lads/ra_joint_predictions')
    
    config = Config()
    logging.info('Command line arguments: %s', sys.argv)

    pretrained_model = sys.argv[1]
    logging.info('Using pretrainde model: %s', pretrained_model)
    
    if pretrained_model == 'None':
        pretrained_model = None
    else:
        pretrained_model = '../trained_models/' + pretrained_model + '.h5'

    model_name = sys.argv[2]
    logging.info('Saving trained model to: %s', model_name)

    # F for feet, H for hands, W for wrists
    joint_type = sys.argv[3]
    # E for Erosion, J for narrowing
    dmg_type = sys.argv[4]

    do_validation = sys.argv[5] == 'Y'

    model_type = sys.argv[6]
    
    is_combined = sys.argv[7] == 'Y'
    
    # load pretrained model
    trained_model, hist_df = train_joints_damage_model(config, model_name, pretrained_model, joint_type, dmg_type, do_validation = do_validation, model_type = model_type, is_combined = is_combined)

    save_pretrained_model(trained_model, 0, '../trained_models/' + model_name)
    hist_df.to_csv('../trained_models/' + model_name + '_hist.csv')
