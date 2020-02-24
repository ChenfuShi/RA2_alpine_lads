import datetime

import tensorflow as tf

from utils.saver import CustomSaver, _get_tensorboard_callback
from dataset.joint_dataset import rsna_joint_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

def pretrain_rnsa_multioutput_model(model_name, config, model_creator):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)

    joint_dataset = rsna_joint_dataset(config).create_rsna_joints_dataset()

    joint_dataset = _split_outcomes(joint_dataset)
    # joint_val_dataset = _split_outcomes(joint_val_dataset)

    model = model_creator(config.joint_img_height, config.joint_img_width)

    model.fit(joint_dataset,
        epochs = 200, steps_per_epoch = 100, verbose = 2, callbacks = [saver, tensorboard_callback])

    return model


def _split_outcomes(dataset, no_joint_types = 13):
    def __split_outcomes(x, y):
        split_y = tf.split(y, [1, 1, no_joint_types], 1)

        return x, (split_y[0], split_y[1], split_y[2])

    return dataset.map(__split_outcomes, num_parallel_calls=AUTOTUNE)

