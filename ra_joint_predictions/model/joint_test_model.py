
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

foot_joints = ['mtp', 'mtp_1', 'mtp_2', 'mtp_3', 'mtp_4', 'mtp_5']

def do_thing():
    joint_inputs = []

    for foot_joint in foot_joints:
        joint_in = keras.layers.Input(shape=[128, 256, 1], name = 'in_' + foot_joint)

        joint_inputs.append(joint_in)

    conv = keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu",input_shape=[128, 256, 1])(joint_inputs)
    # kernel = keras.layers.BatchNormalization()(kernel)
    max_pool = keras.layers.MaxPooling2D((2,2))(conv)
    flat = keras.layers.Flatten()(max_pool)
    dense = keras.layers.Dense(25)(flat)
    output = keras.layers.Dense(5, activation = 'softmax')(dense)
    lam = keras.layers.Lambda(lambda t: K.sum(t * tf.range(5, dtype = tf.float32), axis = -1))(output)

    # out = keras.layers.Dense(6, activation = 'sigmoid')(concat)

    model = keras.models.Model(
        inputs=joint_inputs,
        outputs=[lam],
        name='a_test')

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

    return model