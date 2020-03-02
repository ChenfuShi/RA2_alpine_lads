import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.metrics import top_k_categorical_accuracy

def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)

def argmax_rsme(y_true, y_pred):
    true = tf.cast(K.argmax(y_true), K.floatx())
    pred = tf.cast(K.argmax(y_pred), K.floatx())
    
    return rsme(true, pred)

def softmax_rsme_metric(classes):
    def softmax_rsme(y_true, y_pred):
        true = tf.cast(K.argmax(y_true), K.floatx())
        pred = K.sum(y_pred * classes, axis = 1)
        
        return rsme(true, pred)
    
    return softmax_rsme

def class_softmax_rsme_metric(classes, class_filter):
    def class_softmax_rmse(y_true, y_pred):
        true = tf.cast(K.argmax(y_true), K.floatx())
        pred = K.sum(y_pred * classes, axis = 1)

        idx = tf.where(tf.math.not_equal(tf.cast(class_filter, K.floatx()), true))
 
        true = tf.gather(true, idx)
        pred = tf.gather(pred, idx)

        idx_size = tf.size(idx)

        rsme_val = tf.cond(idx_size == 0,
                    lambda: tf.constant(0, tf.float32),
                    lambda: rsme(true, pred))

        return rsme_val
    
    class_softmax_rmse.__name__ = 'class_softmax_rmse_{}'.format(class_filter)
    
    return class_softmax_rmse
        
def rsme(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))
