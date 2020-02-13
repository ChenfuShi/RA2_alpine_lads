from tensorflow import keras

def create_bigger_kernel_base(input_height, input_width):
    model_kernel = keras.models.Sequential([
        keras.layers.Conv2D(filters = 20, kernel_size=(5,5),activation="relu",input_shape=[input_height, input_width, 1]),
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
        keras.layers.Flatten()
    ])

    return model_kernel