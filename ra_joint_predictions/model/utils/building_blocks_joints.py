from tensorflow import keras

def get_joint_model_input(config):
    return keras.layers.Input(shape = [config.joint_img_height, config.joint_img_width, 1])

def complex_rewritten(input):
    model = _vvg_conv_block(input, 32, 'conv_1')
    model = _vvg_conv_block(model, 32, 'conv_2')
    model = _vvg_conv_block(model, 64, 'conv_3')
    model = _vvg_conv_block(model, 64, 'conv_4')
    model = _vvg_conv_block(model, 128, 'conv_5')
    model = _vvg_conv_block(model, 128, 'conv_6')

    model = keras.layers.Flatten()(model)

    model = _vvg_fc_block(model, 1024, 'fc_1')
    model = _vvg_fc_block(model, 512, 'fc_2')
    model = _vvg_fc_block(model, 256, 'fc_3')

    return model
    
def vvg_joint_model(input):
    model = _vvg_conv_block(input, 16, 'conv_1')
    model = _vvg_conv_block(model, 16, 'conv_2')
    model = _vvg_conv_block(model, 32, 'conv_3')
    model = _vvg_conv_block(model, 32, 'conv_4')
    model = _vvg_conv_block(model, 64, 'conv_5')
    model = _vvg_conv_block(model, 64, 'conv_6')

    model = keras.layers.Flatten()(model)

    model = _vvg_fc_block(model, 512, 'fc_1')
    model = _vvg_fc_block(model, 256, 'fc_2')

    return model

def _vvg_conv_block(input, n_filters, block_prefix):
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), activation = 'relu', padding = 'same',  name = block_prefix + '_conv_1')(input)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_1')(conv)
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), activation = 'relu', padding='same', name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)
    conv = keras.layers.MaxPooling2D((2, 2), name = block_prefix + '_max_pool')(conv)

    return conv

def _vvg_fc_block(input, n_neurons, block_prefix):
    fc = keras.layers.Dense(n_neurons, activation = 'relu', name = block_prefix + '_fc')(input)
    fc = keras.layers.BatchNormalization(name = block_prefix + '_batch')(fc)
    fc = keras.layers.Dropout(0.5, name = block_prefix + '_dropout')(fc)

    return fc

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

def rewritten_complex(config):
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu",padding='same',input_shape=[config.img_height,config.img_width,1]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation="relu",padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024,activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512,activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256,activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),

    ])

    return model


def bigger_kernel_base(config):

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu",input_shape=[config.img_height,config.img_width,1]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512,activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),

    ])

    return model
