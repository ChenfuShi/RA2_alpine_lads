import tensorflow as tf
from tensorflow import keras


def get_feet_model(base_model_joints_loc):

    base_model_joints = keras.models.load_model(base_model_joints_loc, compile = False)
    base_model_joints_no_dense = keras.models.Model(base_model_joints.inputs, base_model_joints.layers[-14].output) 
    base_model_joints_no_dense.trainable = False

    inputs = []
    for i in range(6):
        inputs.append(keras.layers.Input(shape=(224,224,1)))
    outs = []
    for i in range(6):
        outs.append(base_model_joints_no_dense(inputs[i]))

    pred = keras.layers.Concatenate()(outs)
    pred = keras.layers.Dense(1024, activation = "relu")(pred)
    pred = keras.layers.BatchNormalization()(pred)
    pred = keras.layers.Dropout(0.5)(pred)
    pred = keras.layers.Dense(512, activation = "relu")(pred)
    pred = keras.layers.BatchNormalization()(pred)
    pred = keras.layers.Dropout(0.5)(pred)
    pred = keras.layers.Dense(1)(pred)

    combined_model = keras.models.Model(inputs=inputs, outputs=[pred])

    combined_model.compile(optimizer = _get_optimizer(), loss = "mean_absolute_error", metrics = ["mae"])

    return combined_model


def get_hand_model(base_model_joints_loc, base_model_wrist_loc, erosion_flag = False):

    individual_model_joints = keras.models.load_model(base_model_joints_loc, compile = False)
    individual_model_joints_no_dense = keras.models.Model(individual_model_joints.inputs, individual_model_joints.layers[-14].output) 
    individual_model_joints_no_dense.trainable = False
    individual_model_wrist = keras.models.load_model(base_model_wrist_loc, compile = False)
    individual_model_wrist_no_dense = keras.models.Model(individual_model_wrist.inputs, individual_model_wrist.layers[-14].output) 
    individual_model_wrist_no_dense.trainable = False

    inputs = []
    if erosion_flag:
        n_joints = 10
    else:
        n_joints = 9

    for i in range(n_joints + 1):
        inputs.append(keras.layers.Input(shape=(224,224,1)))
    outs = []
    for i in range(n_joints):
        outs.append(individual_model_joints_no_dense(inputs[i]))
    outs.append(individual_model_wrist_no_dense(inputs[-1]))

    pred = keras.layers.Concatenate()(outs)
    pred = keras.layers.Dense(1024, activation = "relu")(pred)
    pred = keras.layers.BatchNormalization()(pred)
    pred = keras.layers.Dropout(0.5)(pred)
    pred = keras.layers.Dense(512, activation = "relu")(pred)
    pred = keras.layers.BatchNormalization()(pred)
    pred = keras.layers.Dropout(0.5)(pred)
    pred = keras.layers.Dense(1)(pred)

    combined_model = keras.models.Model(inputs=inputs, outputs=[pred])

    combined_model.compile(optimizer = _get_optimizer(), loss = "mean_absolute_error", metrics = ["mae"])

    return combined_model


def _get_optimizer():

    lr_decayed_fn = (
      tf.keras.experimental.CosineDecay(initial_learning_rate = 3e-4,
      decay_steps = 100*40,
      alpha=1/3))

    return keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

    #return keras.optimizers.Adam()