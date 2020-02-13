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