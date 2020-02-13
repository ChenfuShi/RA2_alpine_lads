from tensorflow import keras

def create_bigger_kernel_base(inputs):
    conv = keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu")(inputs)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D((2,2))(conv)
    conv = keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters = 40, kernel_size=(5,5),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D((2,2))(conv)
    conv = keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters = 80, kernel_size=(3,3),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D((2,2))(conv)
    conv = keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters = 160, kernel_size=(3,3),activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D((2,2))(conv)
    conv = keras.layers.Flatten()(conv)

    return conv