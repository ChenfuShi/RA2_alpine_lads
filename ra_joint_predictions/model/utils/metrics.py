from tensorflow.keras.metrics import top_k_categorical_accuracy

def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)