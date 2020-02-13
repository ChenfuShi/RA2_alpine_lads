from tensorflow import keras 

def save_pretrained_model(pretrained_model, no_layers_to_remove, model_name):
    new_model = keras.models.Sequential()
        
    # remove the last x layers, by only adding layers before it to the new model
    idx = -1 * no_layers_to_remove
    for layer in pretrained_model.layers[:idx]:
        new_model.add(layer)

    new_model.save(model_name + '.h5') 
