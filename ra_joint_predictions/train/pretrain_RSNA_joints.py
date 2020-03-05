import datetime

import tensorflow as tf

from utils.saver import CustomSaver, _get_tensorboard_callback
from dataset.rsna_joint_dataset import rsna_joint_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

def pretrain_rnsa_multioutput_model(model_name, config, model_creator):
    saver = CustomSaver(model_name, n = 10)
    tensorboard_callback = _get_tensorboard_callback(model_name)

    joint_dataset, joint_val_dataset = rsna_joint_dataset(config).create_rsna_joints_dataset(val_split = True)

    joint_dataset = _split_outcomes(joint_dataset)
    joint_val_dataset = _split_outcomes(joint_val_dataset)

    model = model_creator(config)

    model.fit(joint_dataset,
        epochs = 500, steps_per_epoch = 110, validation_data = joint_val_dataset, validation_steps = 10, verbose = 2, callbacks = [saver, tensorboard_callback])

    return model


def _split_outcomes(dataset, no_joint_types = 13):
    def __split_outcomes(x, y):
        split_y = tf.split(y, [1, 1, no_joint_types], 1)

        return x, (split_y[0], split_y[1], split_y[2])

    return dataset.map(__split_outcomes, num_parallel_calls=AUTOTUNE)


def finetune_model(model,model_name,config,epochs_before=51,epochs_after=201):
    
    tensorboard_callback = _get_tensorboard_callback(model_name)

    joint_dataset, joint_val_dataset = rsna_joint_dataset(config).create_rsna_joints_dataset(val_split = True)
    joint_dataset = _split_outcomes(joint_dataset,10)
    joint_val_dataset = _split_outcomes(joint_val_dataset,10)

    saver = CustomSaver(model_name + "before", n = 10)
    model.fit(joint_dataset,
    epochs = epochs_before, steps_per_epoch = 1000, validation_data = joint_val_dataset, validation_steps = 10, verbose = 2, callbacks = [saver, tensorboard_callback])

    for layer in model.layers:
        layer.trainable = True

    # need to recompile after trainable
    losses = {
        'boneage_pred': 'mean_squared_error',
        'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': 'categorical_crossentropy',
    }

    lossWeights = {'boneage_pred': 0.005, 'sex_pred': 2, 'joint_type_pred': 1}

    model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'categorical_accuracy'})


    saver = CustomSaver(model_name + "after", n = 10)
    model.fit(joint_dataset,
    epochs = epochs_after, steps_per_epoch = 1000, validation_data = joint_val_dataset, validation_steps = 10, verbose = 2, callbacks = [saver, tensorboard_callback])
