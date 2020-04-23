from tensorflow import keras

elu_activation = lambda x: keras.activations.elu(x, alpha = 0.3)

def get_joint_model_input(config):
    return keras.layers.Input(shape = [config.joint_img_height, config.joint_img_width, 1])

def complex_rewritten(input, initializer = 'he_uniform', decay = 1e-4):
    if decay is not None:
        kernel_regularizer = keras.regularizers.l2(decay)
    else:
        kernel_regularizer = None
    
    model = _vvg_conv_block(input, 32, 'conv_1', initializer, kernel_regularizer)
    model = _vvg_conv_block(model, 32, 'conv_2', initializer, kernel_regularizer)
    model = _vvg_conv_block(model, 64, 'conv_3', initializer, kernel_regularizer)
    model = _vvg_conv_block(model, 64, 'conv_4', initializer, kernel_regularizer)
    model = _vvg_conv_block(model, 128, 'conv_5', initializer, kernel_regularizer)
    model = _vvg_conv_block(model, 128, 'conv_6', initializer, kernel_regularizer)

    model = keras.layers.Flatten()(model)

    model = _vvg_fc_block(model, 1024, 'fc_1', initializer = initializer, kernel_regularizer = None)
    model = _vvg_fc_block(model, 512, 'fc_2', initializer = initializer, kernel_regularizer = None)
    model = _vvg_fc_block(model, 256, 'fc_3', initializer = initializer, kernel_regularizer = None)

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

def _vvg_conv_block(input, n_filters, block_prefix, initializer, kernel_regularizer):
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_1')(input)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_1')(conv)
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding='same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)
    conv = keras.layers.MaxPooling2D((2, 2), name = block_prefix + '_max_pool')(conv)

    return conv

def _vvg_fc_block(input, n_neurons, block_prefix, initializer = 'glorot_uniform', kernel_regularizer = None):
    fc = keras.layers.Dense(n_neurons, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_fc')(input)
    fc = keras.layers.ReLU()(fc)
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

def rewritten_elu(config):
    elu_activation = lambda x: keras.activations.elu(x, alpha = 0.1)
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation=elu_activation,padding='same',input_shape=[config.img_height,config.img_width,1]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 64, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters = 128, kernel_size=(3,3),activation=elu_activation,padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024,activation=elu_activation),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512,activation=elu_activation),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256,activation=elu_activation),
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

def elu_res_net(input, alpha = 1.0):
    model = _res_head(input)
    
    model = _elu_res_stack(model, 32, 3, alpha)
    model = _elu_res_stack(model, 64, 4, alpha)
    model = _elu_res_stack(model, 128, 5, alpha)
    model = _elu_res_stack(model, 256, 3, alpha, last_stride = 1)

    model = keras.layers.ELU(alpha = alpha)(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.GlobalAveragePooling2D()(model)
    
    return model

def _res_head(input):
    head = keras.layers.ZeroPadding2D(padding = 3)(input)
    head = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, padding = 'valid', kernel_initializer = 'he_uniform', kernel_regularizer = keras.regularizers.l2(5e-4))(head)
    head = keras.layers.ZeroPadding2D(padding = 1)(head)
    head = keras.layers.MaxPool2D(3, strides = 2)(head)
    
    return head

def _elu_res_stack(input, no_filters, no_blocks, alpha, last_stride = 2):
    conv = _elu_res_block(input, no_filters, alpha, use_conv = True)
    
    for n in range(2, no_blocks):
        conv = _elu_res_block(conv, no_filters, alpha)
        
    conv = _elu_res_block(conv, no_filters, alpha, last_stride = last_stride)
    
    return conv

def _elu_res_block(input, no_filters, alpha, use_conv = False, last_stride = 1):
    # Create the shortcut connection
    # First shortcut uses a conv to update the number of channels to the expected output of the res block
    if use_conv:
        shortcut = keras.layers.Conv2D(filters = no_filters, kernel_size = 1, strides = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = keras.regularizers.l2(5e-4))(input)
    else:
        # The last res block halfs the number of channels via stride 2, so the shortcut must do the same
        if last_stride > 1:
            shortcut = keras.layers.MaxPool2D(1, strides = last_stride, padding = 'same')(input)
        else:
            shortcut = input
            
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, strides = last_stride, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = keras.regularizers.l2(5e-4))(input)
    conv = keras.layers.ELU(alpha = alpha)(conv)
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = keras.regularizers.l2(5e-4))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Add()([shortcut, conv])
    
    return conv

def extended_complex(input, initializer = 'he_uniform', decay = 5e-4):
    model = _extended_vgg_complex_head(input, initializer, decay)
    
    model = _extended_vgg_complex_conv_block(model, 32, 'conv_block_1', initializer, decay)
    model = _extended_vgg_complex_conv_block(model, 32, 'conv_block_2', initializer, decay, double_last = True)
    model = keras.layers.MaxPooling2D((2, 2), name = 'max_pool_1')(model)
    model = _extended_vgg_complex_conv_block(model, 64, 'conv_block_3', initializer, decay)
    model = _extended_vgg_complex_conv_block(model, 64, 'conv_block_4', initializer, decay, double_last = True)
    model = keras.layers.MaxPooling2D((2, 2), name = 'max_pool_2')(model)
    model = _extended_vgg_complex_conv_block(model, 128, 'conv_block_5', initializer, decay)
    model = _extended_vgg_complex_conv_block(model, 128, 'conv_block_6', initializer, decay, double_last= True)
    model = keras.layers.MaxPooling2D((2, 2), name = 'max_pool_3')(model)
    
    model = keras.layers.Flatten()(model)

    # model = _vvg_fc_block(model, 1024, 'fc_1', initializer = initializer)
    model = _vvg_fc_block(model, 512, 'fc_1', initializer = initializer)
    model = _vvg_fc_block(model, 256, 'fc_2', initializer = initializer)
    
    return model
    
def _extended_vgg_complex_head(input, initializer, decay):
    kernel_regularizer = keras.regularizers.l2(decay)
    
    head = keras.layers.ZeroPadding2D(padding = ((3, 3), (3, 3)))(input)
    head = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = 'head_conv')(head)
    head = keras.layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(head)
    conv = keras.layers.ReLU()(head)
    conv = keras.layers.BatchNormalization(name = 'head_batch_norm')(head)
    head = keras.layers.MaxPool2D(3, strides = 2, name = 'head_max_pool')(head)
    
    return head
    
def _extended_vgg_complex_conv_block(input, n_filters, block_prefix, initializer, decay, double_last = False):
    last_n = n_filters if not double_last else 2 * n_filters
    
    kernel_regularizer = keras.regularizers.l2(decay)
    
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = 1, use_bias = False, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_norm_conv')(input)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_norm_conv_batch')(conv)

    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding=  'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_1')(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_1')(conv)

    conv = keras.layers.Conv2D(filters = last_n, kernel_size = (3, 3), padding = 'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)

    return conv

def avg_joint_vgg(input, initializer = 'he_uniform', decay = 1e-5):
    if decay is not None:
        kernel_regularizer = keras.regularizers.l2(decay)
    else:
        kernel_regularizer = None
    
    # model = _avg_joint_vgg_head(input, initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(input, 32, 'conv_block_1', initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(model, 32, 'conv_block_2', initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(model, 64, 'conv_block_3', initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(model, 64, 'conv_block_4', initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(model, 128, 'conv_block_5', initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(model, 128, 'conv_block_6', initializer, kernel_regularizer)
    model = _avg_joint_vgg_conv_block(model, 256, 'conv_block_7', initializer, kernel_regularizer)
    
    model = keras.layers.GlobalAveragePooling2D()(model)
    
    return model

def _avg_joint_vgg_head(input, initializer, kernel_regularizer):
    head = keras.layers.ZeroPadding2D(padding = ((3, 3), (3, 3)))(input)
    head = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = 'head_conv')(head)
    head = keras.layers.BatchNormalization(name = 'head_batch_norm')(head)
    head = keras.layers.ReLU()(head)
    head = keras.layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(head)
    head = keras.layers.AveragePooling2D(3, strides = 2, name = 'head_avg_pool')(head)
    
    return head

def _avg_joint_vgg_conv_block(input, n_filters, block_prefix, initializer, kernel_regularizer):
    
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding=  'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_1')(input)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_1')(conv)
    conv = keras.layers.ReLU()(conv)
    
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding=  'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)
    conv = keras.layers.ReLU()(conv)
    
    conv = keras.layers.AveragePooling2D()(conv)
    
    return conv
    
