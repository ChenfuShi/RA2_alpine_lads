import datetime

import tensorflow as tf
import tensorflow.keras.backend as K

from utils.saver import CustomSaver, _get_tensorboard_callback
from dataset.rsna_joint_dataset import rsna_joint_dataset
from keras_adamw import AdamW
from train.utils.callbacks import AdamWWarmRestartCallback

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


def finetune_model(model,model_name,joint_dataset, joint_val_dataset ,epochs_before=51,epochs_after=201, n_outputs = 10, is_wrists = False):
    joint_dataset = _split_outcomes(joint_dataset,n_outputs)
    joint_val_dataset = _split_outcomes(joint_val_dataset,n_outputs)

    tensorboard_callback = _get_tensorboard_callback(model_name, log_dir = 'logs/tensorboard_RSNA/')

    if epochs_before > 0:
        saver = CustomSaver(model_name + "before", n = 10)
        model.fit(joint_dataset,
            epochs = epochs_before, steps_per_epoch = 1750, validation_data = joint_val_dataset, validation_steps = 175, verbose = 2, callbacks = [saver, tensorboard_callback])

    for layer in model.layers:
        layer.trainable = True

    # need to recompile after trainable
    losses = {
        'boneage_pred': 'mean_squared_error',
        'sex_pred' : 'binary_crossentropy',
        'joint_type_pred': 'categorical_crossentropy',
    }

    lossWeights = {'boneage_pred': 0.005, 'sex_pred': 2, 'joint_type_pred': 1}
    
    optimizer = _get_optimizier(model)
    
    model.compile(optimizer = optimizer, loss = losses, loss_weights = lossWeights, 
        metrics={'boneage_pred': 'mae', 'sex_pred': 'binary_accuracy', 'joint_type_pred': 'categorical_accuracy'})

    
    def scheduler(epoch):
        if epoch < 20:
            return 1e-2
        elif epoch < 50:
            return 5e-3
        elif epoch < 80:
            return 5e-4
        else:
            return 5e-5
        
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    # adamW_warm_restart_callback = AdamWWarmRestartCallback(restart_epochs = 25)

    saver = CustomSaver(model_name + "after", n = 10)
    
    steps_per_epoch = 1750
    val_steps = 175

    if is_wrists:
        steps_per_epoch = 135
        val_steps = 14
    
    model.fit(joint_dataset,
        epochs = epochs_after, steps_per_epoch = steps_per_epoch, validation_data = joint_val_dataset, validation_steps = val_steps, verbose = 2, callbacks = [saver, tensorboard_callback])
    
def _get_optimizier(model):
    weight_decays = {}

    # for layer in model.layers:
        # layer.kernel_regularizer = tf.keras.regularizers.l2(5e-4)
        # weight_decays.update({layer.name: 1e-6})

    #ptimizer = AdamW(lr=3e-4, weight_decays = weight_decays, use_cosine_annealing = True, total_iterations = 1750 * 25, init_verbose = False)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4)
    
    return optimizer