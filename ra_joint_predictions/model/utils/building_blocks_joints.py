import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from tensorflow import keras

elu_activation = lambda x: keras.activations.elu(x, alpha = 0.3)

def get_joint_model_input(config):
    return keras.layers.Input(shape = [config.joint_img_height, config.joint_img_width, 1])

def complex_rewritten(input, initializer = 'he_uniform', decay = 1e-4, use_dense = True):
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

    if use_dense:
        model = keras.layers.Flatten()(model)

        model = _vvg_fc_block(model, 1024, 'fc_1', initializer = initializer, kernel_regularizer = None)
        model = _vvg_fc_block(model, 512, 'fc_2', initializer = initializer, kernel_regularizer = None)
        model = _vvg_fc_block(model, 256, 'fc_3', initializer = initializer, kernel_regularizer = None)
    else:
        model = keras.layers.GlobalAveragePooling2D()(model)

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

def _vvg_fc_block(input, n_neurons, block_prefix, initializer = 'glorot_uniform', kernel_regularizer = None, use_group_norm = False, use_dropout = False, use_renorm = False):
    fc = keras.layers.Dense(n_neurons, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_fc')(input)
    fc = keras.layers.ReLU(name = block_prefix + '_relu')(fc)
    
    if use_group_norm:
        fc = tfa.layers.GroupNormalization(groups = 32, name = f'{block_prefix}_batch')(fc)
    else:
        fc = keras.layers.BatchNormalization(renorm = use_renorm, name = block_prefix + '_batch')(fc)
    
    if use_dropout:
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

def relu_joint_res_net(input, kernel_initializer = 'he_uniform', decay = 1e-4):
    if decay is not None:
        kernel_regularizer = keras.regularizers.l2(decay)
    else:
        kernel_regularizer = None
    
    model = _res_head(input, kernel_initializer, kernel_regularizer)
    
    model = _elu_res_stack(model, 32, 3, kernel_initializer, kernel_regularizer)
    model = _elu_res_stack(model, 64, 4, kernel_initializer, kernel_regularizer)
    model = _elu_res_stack(model, 128, 5, kernel_initializer, kernel_regularizer)
    model = _elu_res_stack(model, 256, 3, kernel_initializer, kernel_regularizer, last_stride = 1)

    model = keras.layers.ReLU()(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.GlobalAveragePooling2D()(model)
    
    return model

def _res_head(input, kernel_initializer, kernel_regularizer):
    head = keras.layers.ZeroPadding2D(padding = 3)(input)
    head = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, padding = 'valid', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(head)
    head = keras.layers.ReLU()(head)
    head = keras.layers.BatchNormalization()(head)
    head = keras.layers.ZeroPadding2D(padding = 1)(head)
    head = keras.layers.MaxPool2D(3, strides = 2)(head)
    
    return head

def _elu_res_stack(input, no_filters, no_blocks, kernel_initializer, kernel_regularizer, last_stride = 2):
    conv = _elu_res_block(input, no_filters, kernel_initializer, kernel_regularizer, use_conv = True)
    
    for n in range(2, no_blocks):
        conv = _elu_res_block(conv, no_filters, kernel_initializer, kernel_regularizer)
        
    conv = _elu_res_block(conv, no_filters, kernel_initializer, kernel_regularizer, last_stride = last_stride)
    
    return conv

def _elu_res_block(input, no_filters, kernel_initializer, kernel_regularizer, use_conv = False, last_stride = 1):
    # Create the shortcut connection
    # First shortcut uses a conv to update the number of channels to the expected output of the res block
    if use_conv:
        shortcut = keras.layers.Conv2D(filters = no_filters, kernel_size = 1, strides = 1, padding = 'same', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(input)
        shortcut = keras.layers.ReLU()(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    else:
        # The last res block halfs the number of channels via stride 2, so the shortcut must do the same
        if last_stride > 1:
            shortcut = keras.layers.MaxPool2D(1, strides = last_stride, padding = 'same')(input)
        else:
            shortcut = input
            
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, strides = last_stride, padding = 'same', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(input)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, padding = 'same', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Add()([shortcut, conv])
    
    return conv

def extended_complex(input, initializer = 'he_uniform', decay = 5e-4):
    if decay is not None:
        kernel_regularizer = keras.regularizers.l2(decay)
    else:
        kernel_regularizer = None
    
    model = _extended_vgg_complex_head(input, initializer, kernel_regularizer)
    
    model = _extended_vgg_complex_conv_block(model, 32, 'conv_block_1', initializer, kernel_regularizer)
    model = _extended_vgg_complex_conv_block(model, 32, 'conv_block_2', initializer, kernel_regularizer)
    model = _extended_vgg_complex_conv_block(model, 64, 'conv_block_3', initializer, kernel_regularizer)
    model = _extended_vgg_complex_conv_block(model, 64, 'conv_block_4', initializer, kernel_regularizer)
    model = _extended_vgg_complex_conv_block(model, 128, 'conv_block_5', initializer, kernel_regularizer)
    model = _extended_vgg_complex_conv_block(model, 128, 'conv_block_6', initializer, kernel_regularizer, double_last= True)
    
    model = keras.layers.Flatten()(model)

    model = _vvg_fc_block(model, 1024, 'fc_1', initializer = initializer)
    model = _vvg_fc_block(model, 512, 'fc_2', initializer = initializer)
    model = _vvg_fc_block(model, 256, 'fc_3', initializer = initializer)
    
    return model
    
def _extended_vgg_complex_head(input, initializer, kernel_regularizer):
    head = keras.layers.ZeroPadding2D(padding = ((3, 3), (3, 3)))(input)
    head = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = 'head_conv')(head)
    head = keras.layers.ReLU()(head)
    head = keras.layers.BatchNormalization(name = 'head_batch_norm')(head)
    head = keras.layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(head)
    head = keras.layers.MaxPool2D(3, strides = 2, name = 'head_max_pool')(head)
    
    return head
    
def _extended_vgg_complex_conv_block(input, n_filters, block_prefix, initializer, kernel_regularizer, double_last = False):
    last_n = n_filters if not double_last else 2 * n_filters
    
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = 1, use_bias = False, kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_norm_conv')(input)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_norm_conv_batch')(conv)

    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding=  'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_1')(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_1')(conv)

    conv = keras.layers.Conv2D(filters = last_n, kernel_size = (3, 3), padding = 'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)
    
    if not double_last:
        conv = keras.layers.MaxPooling2D((2, 2), name = block_prefix + 'max_pool')(conv)

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
    
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = initializer, kernel_regularizer = kernel_regularizer, name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.BatchNormalization(name = block_prefix + '_batch_2')(conv)
    conv = keras.layers.ReLU()(conv)
    
    conv = keras.layers.AveragePooling2D()(conv)
    
    return conv

def vgg19_with_sp_dropout(input):
    #head = keras.layers.ZeroPadding2D(padding = ((3, 3), (3, 3)), name = 'head_pad')(input)
    #head = keras.layers.Conv2D(filters = 16, kernel_size = (7, 7), strides = 2, padding = 'valid', kernel_initializer = 'he_uniform', name = 'head_conv')(head)
    #head = keras.layers.ReLU(name = 'head_relu')(head)
    #head = keras.layers.BatchNormalization(name = 'head_batch_norm')(head)
    #head = keras.layers.ZeroPadding2D(padding=(1, 1), name='head_pool_pad')(head)
    #head = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name = 'head_pool')(head)
    # head = keras.layers.SpatialDropout2D(0.25, name = 'head_dropout')(head)
    
    conv = _vgg_block_with_sp(input, 2, 32, 'conv_block_1', add_bottleneck = True)
    conv = _vgg_block_with_sp(conv, 2, 32, 'conv_block_2', add_bottleneck = True)
    conv = _vgg_block_with_sp(conv, 4, 64, 'conv_block_3', add_bottleneck = True)
    conv = _vgg_block_with_sp(conv, 4, 64, 'conv_block_4', add_bottleneck = True)
    conv = _vgg_block_with_sp(conv, 4, 128, 'conv_block_5', add_bottleneck = True, max_pool = False)
    # conv = _vgg_block_with_sp(conv, 4, 128, 'conv_block_6', max_pool = False, add_bottleneck = True)
    # conv = _vgg_block_with_sp(conv, 4, 128, 'conv_block_6', max_pool = False)
    # conv = _vgg_block_with_sp(conv, 4, 128, 'conv_block_6', max_pool = False)
    # conv = _vgg_block_with_sp(conv, 4, 128, 'conv_block_7', max_pool = False)
    # conv = _vgg_block_with_sp(conv, 4, 256, 'conv_block_7', max_pool = False)
    # conv = _vgg_block_with_sp(conv, 4, 128, 'conv_block_7', max_pool = False)
    
    #conv = keras.layers.SpatialDropout2D(0.5, name = f'dropout')(conv)
    
    #conv = _1x1_convs(conv, 64, '1x1_conv_block_1')
    #conv = _1x1_convs(conv, 32, '1x1_conv_block_2')
    #conv = _1x1_convs(conv, 32, '1x1_conv_block_3')
    
    model = keras.layers.GlobalAveragePooling2D()(conv)
    # model = keras.layers.Dropout(0.5)(model)
    # model = _vvg_fc_block(model, 64, 'fc_1', initializer = 'he_uniform', kernel_regularizer = None, use_group_norm = True)
    # model = _vvg_fc_block(model, 32, 'fc_2', initializer = 'he_uniform', kernel_regularizer = None, use_group_norm = True)
    
    return model
    
def _vgg_block_with_sp(input, n_convs, n_filters, block_prefix, max_pool = True, add_bottleneck = False):
    conv = input
    
    for n in range(n_convs):
        conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = f'{block_prefix}_conv{n}_conv')(conv)
        conv = keras.layers.ReLU(name = f'{block_prefix}_conv{n}_relu')(conv)
        conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_conv{n}_batch')(conv)

    if add_bottleneck:   
        conv = keras.layers.Conv2D(filters = int(n_filters / 2), kernel_size = 1, kernel_initializer = 'he_uniform', name = f'{block_prefix}_bottleneck_conv')(conv)
        conv = keras.layers.ReLU(name = f'{block_prefix}_bottleneck_relu')(conv)
        # conv = tfa.layers.GroupNormalization(groups = 16, name = f'{block_prefix}_bottleneck_batch')(conv)
    
    if max_pool:
        conv = keras.layers.MaxPooling2D((2, 2), name = f'{block_prefix}_max_pool')(conv)
    
    # conv = keras.layers.SpatialDropout2D(0.25, name = f'{block_prefix}_dropout')(conv)
        
    return conv

def _1x1_convs(input, n_filters, block_prefix):
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (1, 1), kernel_initializer = 'he_uniform', name = f'{block_prefix}_1x1_conv')(input)
    conv = keras.layers.ReLU(name = f'{block_prefix}_1x1_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_1x1_batch')(conv)
    
    return conv

def small_vgg_with_bottleneck(input):
    conv = _bottlnecked_vvg_conv_block(input, 32, 'conv_1')
    conv = _bottlnecked_vvg_conv_block(conv, 32, 'conv_2')
    conv = _bottlnecked_vvg_conv_block(conv, 64, 'conv_3')
    conv = _bottlnecked_vvg_conv_block(conv, 64, 'conv_4')
    conv = _bottlnecked_vvg_conv_block(conv, 128, 'conv_5')
    conv = _bottlnecked_vvg_conv_block(conv, 128, 'conv_6', add_pool = False)
    
    model = keras.layers.GlobalAveragePooling2D()(conv)
    
    return model

def _bottlnecked_vvg_conv_block(input, n_filters, block_prefix, add_pool = True):
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = block_prefix + '_conv_1')(input)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = block_prefix + '_batch_1')(conv)
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding='same', kernel_initializer = 'he_uniform', name = block_prefix + '_conv_2')(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = block_prefix + '_batch_2')(conv)
    
    if add_pool:
        conv = _bottle_neck(conv, int(K.int_shape(conv)[3] * 0.5), f'{block_prefix}_bottleneck', add_activation = True, add_norm = False)
        
        conv = keras.layers.MaxPooling2D((2, 2), name = block_prefix + '_max_pool')(conv)

    return conv

def small_resnet_with_bottleneck(input):
    conv = _res_stack(input, 2, 16, 'res_stack_1')
    conv = _res_stack(conv, 3, 16, 'res_stack_2')
    conv = _res_stack(conv, 5, 32, 'res_stack_3')
    conv = _res_stack(conv, 8, 32, 'res_stack_4')
    conv = _res_stack(conv, 8, 64, 'res_stack_5')
    conv = _res_stack(conv, 5, 64, 'res_stack_6', last_stride = 1)
    # conv = _res_stack(conv, 4, 128, 'res_stack_7')
    #conv = _res_stack(conv, 4, 128, 'res_stack_8', last_stride = 1)
    
    model = keras.layers.GlobalAveragePooling2D()(conv)
    
    return model
    
def _res_stack(input, no_blocks, no_filters, block_prefix, last_stride = 2):
    conv = _res_block(input, no_filters, block_prefix + '_conv_block', use_conv = True)
    
    for n in range(2, no_blocks):
        conv = _res_block(conv, no_filters, f'{block_prefix}_id_{n}_block')
        
    conv = _res_block(conv, no_filters, block_prefix + '_end_block', last_stride = last_stride)

    # if last_stride > 1:
        # conv = _bottle_neck(conv, int(no_filters / 2), block_prefix)
    
    return conv

def _res_block(input, no_filters, block_prefix, kernel_initializer = 'he_uniform', use_conv = False, last_stride = 1):
    # Create the shortcut connection
    # First shortcut uses a conv to update the number of channels to the expected output of the res block
    if use_conv:
        shortcut = keras.layers.Conv2D(filters = int(no_filters / 2), kernel_size = 1, strides = 1, padding = 'same', kernel_initializer = kernel_initializer, name = f'{block_prefix}_skip_conv')(input)
        shortcut = keras.layers.ReLU(name = f'{block_prefix}_skip_relu')(shortcut)
        shortcut = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_skip_batch')(shortcut)
    else:
        # The last res block halfs the number of channels via stride 2, so the shortcut must do the same
        if last_stride > 1:
            shortcut = keras.layers.MaxPool2D(1, strides = last_stride, padding = 'same', name = f'{block_prefix}_skip_half')(input)
        else:
            shortcut = input
            
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, strides = last_stride, padding = 'same', kernel_initializer = kernel_initializer, name = f'{block_prefix}_conv')(input)
    conv = keras.layers.ReLU(name = f'{block_prefix}_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_batch')(conv)
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, padding = 'same', kernel_initializer = kernel_initializer, name = f'{block_prefix}_conv2')(conv)
    conv = keras.layers.ReLU(name = f'{block_prefix}_relu2')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_batch2')(conv)
    
    conv = _bottle_neck(conv, int(K.int_shape(conv)[3] * 0.5), f'{block_prefix}_bottleneck', add_activation = False, add_norm = False)
    
    conv = keras.layers.Add()([shortcut, conv])
    conv = keras.layers.ReLU(name = f'{block_prefix}_sum_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_sum_batch')(conv)
    
    return conv

def small_densenet(input):
    n_filters = 16
    
    conv = keras.layers.Conv2D(filters = 2 * n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = 'head_conv')(input)
    conv = keras.layers.ReLU(name = 'head_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = 'head_batch')(conv)
    conv = keras.layers.AveragePooling2D((2, 2), name = 'head_avg_pool')(conv)
    
    conv = _dense_block(conv, n_filters, 2, 'dense_block_1')
    conv = _dense_block(conv, n_filters, 3, 'dense_block_2')
    conv = _dense_block(conv, n_filters, 5, 'dense_block_3')
    conv = _dense_block(conv, n_filters, 5, 'dense_block_4')
    conv = _dense_block(conv, n_filters, 3, 'dense_block_5', add_pool = False)
    
    model = keras.layers.GlobalAveragePooling2D()(conv)
    
    return model

def _dense_block(input, n_filters, n_convs, block_prefix, add_pool = True, add_bottle_neck = False):
    conv = _dense_conv_group(input, n_filters, f'{block_prefix}_conv_block_0', is_first = True, add_bottle_neck = add_bottle_neck)
    
    for n in range(2, n_convs+1):
        conv = _dense_conv_group(conv, n_filters, f'{block_prefix}_conv_block_{n}', add_bottle_neck = add_bottle_neck)
    
    # ID Block
    conv = keras.layers.ReLU(name = f'{block_prefix}_last_concat_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_last_concat_batch')(conv)
    
    if not add_bottle_neck:
        conv = _bottle_neck(conv, int(K.int_shape(conv)[3] * 0.5), block_prefix)
    
    if add_pool:
        conv = keras.layers.AveragePooling2D((2, 2), name = block_prefix + 'avg_pool')(conv)
    
    return conv
    
def _dense_conv_group(input, no_filters, block_prefix, is_first = False, add_bottle_neck = False):
    if not is_first:
        conv = keras.layers.ReLU(name = f'{block_prefix}_concat_relu')(input)
        conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_concat_batch')(conv)
    else:
        conv = input

    conv = keras.layers.Conv2D(filters = 4 * no_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', name = f'{block_prefix}_conv1')(conv)
    conv = keras.layers.ReLU(name = f'{block_prefix}_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_batch')(conv)
    
    conv = keras.layers.Conv2D(filters = no_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', name = f'{block_prefix}_conv2')(conv)
    
    if add_bottle_neck:
        conv = keras.layers.ReLU(name = f'{block_prefix}_relu2')(conv)
        conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_batch2')(conv)
        
        conv = _bottle_neck(conv, int(K.int_shape(conv)[3] * 0.5), block_prefix, add_activation = False)
        
    conv = keras.layers.Concatenate(name = f'{block_prefix}_concat')([input, conv])
    
    return conv

def bottlenecked_small_dense(input):
    n_filters = 16
    
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = 'head_conv')(input)
    conv = keras.layers.ReLU(name = 'head_relu')(conv)
    conv = keras.layers.BatchNormalization(renorm = True, name = 'head_batch')(conv)
    
    conv = _bottle_neck(conv, int(n_filters / 2), 'head', add_norm = True)
    
    conv = keras.layers.AveragePooling2D((2, 2), name = 'head_avg_pool')(conv)
    
    conv = _dense_block(conv, n_filters, 2, 'dense_block_1', add_bottle_neck = True)
    conv = _dense_block(conv, n_filters, 3, 'dense_block_2', add_bottle_neck = True)
    conv = _dense_block(conv, n_filters, 5, 'dense_block_3', add_bottle_neck = True)
    conv = _dense_block(conv, n_filters, 8, 'dense_block_4', add_bottle_neck = True, add_pool = False)
    # conv = _dense_block(conv, n_filters, 3, 'dense_block_5', add_bottle_neck = True, add_pool = False)
    
    model = keras.layers.GlobalAveragePooling2D()(conv)
    
    return model

def _bottle_neck(input, n_filters, block_prefix, add_activation = True, add_norm = False):
    conv = keras.layers.Conv2D(filters = n_filters, kernel_size = 1, kernel_initializer = 'he_uniform', name = f'{block_prefix}_bottleneck_conv')(input)
    
    if add_activation:
        conv = keras.layers.ReLU(name = f'{block_prefix}_bottleneck_relu')(conv)
        
    if add_norm:
        conv = keras.layers.BatchNormalization(renorm = True, name = f'{block_prefix}_bottleneck_batch')(conv)
    
    return conv
