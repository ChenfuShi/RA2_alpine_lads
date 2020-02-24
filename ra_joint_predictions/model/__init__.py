from tensorflow import keras

def create_bigger_kernel_base(inputs):
    conv_1 = keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu")(inputs)
    conv_2 = keras.layers.BatchNormalization()(conv_1)
    conv_3 = keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu")(conv_2)
    conv_4 = keras.layers.BatchNormalization()(conv_3)
    conv_5 = keras.layers.MaxPooling2D((2,2))(conv_4)
    conv_6 = keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu")(conv_5)
    conv_7 = keras.layers.BatchNormalization()(conv_6)
    conv_8 = keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu")(conv_7)
    conv_9 = keras.layers.BatchNormalization()(conv_8)
    conv_10 = keras.layers.MaxPooling2D((2,2))(conv_9)
    conv_11 = keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu")(conv_10)
    conv_12 = keras.layers.BatchNormalization()(conv_11)
    conv_13 = keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu")(conv_12)
    conv_14= keras.layers.BatchNormalization()(conv_13)
    conv_15 = keras.layers.MaxPooling2D((2,2))(conv_14)
    conv_16 = keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu")(conv_15)
    conv_17 = keras.layers.BatchNormalization()(conv_16)
    conv_18 = keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu")(conv_17)
    conv_19 = keras.layers.BatchNormalization()(conv_18)
    conv_20 = keras.layers.MaxPooling2D((2,2))(conv_19)
    conv_21 = keras.layers.Flatten()(conv_20)

    return conv_21

def create_complex_joint_model(input):
    model = _conv_block(input, 32, 'conv_1')
    model = _conv_block(model, 32, 'conv_2')
    model = _conv_block(model, 64, 'conv_3')
    model = _conv_block(model, 64, 'conv_4')
    model = _conv_block(model, 128, 'conv_5')
    model = _conv_block(model, 128, 'conv_6')

    model = keras.layers.Flatten()(model)

    model = _fc_block(model, 1024, 'fc_1')
    model = _fc_block(model, 512, 'fc_2')
    model = _fc_block(model, 256, 'fc_3')

    return model

def _conv_block(input, n_filters, block_prefix):
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding='same', name = block_prefix + '_conv_1')(input)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_1')(conv)
    conv = keras.layers.Activation('relu', name = block_prefix + '_act_1')(conv)
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding='same', name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)
    conv = keras.layers.Activation('relu', name = block_prefix + '_act_2')(conv)
    conv = keras.layers.MaxPooling2D((2, 2), name = block_prefix + '_max_pool')(conv)

    return conv

def _fc_block(input, n_neurons, block_prefix):
    fc = keras.layers.Dense(n_neurons, name = block_prefix + '_fc')(input)
    fc = keras.layers.BatchNormalization(name = block_prefix + '_batch')(fc)
    fc = keras.layers.Activation('relu', name = block_prefix + '_act')(fc)
    fc = keras.layers.Dropout(0.5, name = block_prefix + '_dropout')(fc)

    return fc