from tensorflow import keras


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