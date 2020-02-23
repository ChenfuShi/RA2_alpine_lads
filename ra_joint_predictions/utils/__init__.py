from tensorflow import keras
from tensorflow.keras.metrics import top_k_categorical_accuracy

def save_pretrained_model(pretrained_model, no_layers_to_remove, model_name):
    if no_layers_to_remove > 0:
        new_model = keras.models.Sequential()
        
        # remove the last x layers, by only adding layers before it to the new model
        idx = -1 * no_layers_to_remove
        for layer in pretrained_model.layers[:idx]:
            new_model.add(layer)

        new_model.save(model_name + '.h5')
    else:
        pretrained_model.save(model_name + '.h5')

def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)