import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.metrics import top_k_categorical_accuracy

def brier_score(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)

def argmax_rmse(y_true, y_pred):
    true = tf.cast(K.argmax(y_true), K.floatx())
    pred = tf.cast(K.argmax(y_pred), K.floatx())
    
    return _rmse(true, pred)

def softmax_mae_metric(classes):
    def softmax_mae(y_true, y_pred):
        true = tf.cast(K.argmax(y_true), K.floatx())
        pred = K.sum(y_pred * classes, axis = 1)
        
        return _mae(true, pred)
    
    softmax_mae.__name__ = 'mae'
    
    return softmax_mae

def softmax_rmse_metric(classes):
    def softmax_rmse(y_true, y_pred):
        true = tf.cast(K.argmax(y_true), K.floatx())
        pred = K.sum(y_pred * classes, axis = 1)
        
        return _rmse(true, pred)
    
    softmax_rmse.__name__ = 'rmse'
    
    return softmax_rmse

def class_filter_softmax_rmse_metric(classes, class_filter):
    def class_filter_softmax_rmse(y_true, y_pred):
        true = tf.cast(K.argmax(y_true), K.floatx())
        pred = K.sum(y_pred * classes, axis = 1)

        idx = tf.where(tf.math.not_equal(tf.cast(class_filter, K.floatx()), true))
 
        true = tf.gather(true, idx)
        pred = tf.gather(pred, idx)

        idx_size = tf.size(idx)

        rmse_val = tf.cond(idx_size == 0,
                    lambda: tf.constant(0, tf.float32),
                    lambda: _rmse(true, pred))

        return rmse_val
    
    class_filter_softmax_rmse.__name__ = 'filter_{}_rmse'.format(class_filter)
    
    return class_filter_softmax_rmse
    
# RMSE for Regression Task
def rmse_metric(max_outcome, offset_factor = 1):
    def rmse(y_true, y_pred):
        y_pred = y_pred * offset_factor
        
        y_pred = K.maximum(y_pred, tf.constant(0, tf.float32))
        y_pred = K.minimum(y_pred, tf.constant(max_outcome, tf.float32))
    
        return _rmse(y_true, y_pred)
    
    return rmse

def mae_metric(max_outcome, offset_factor = 1):
    def mae(y_true, y_pred):
        y_pred = y_pred * offset_factor
        
        y_pred = K.maximum(y_pred, tf.constant(0, tf.float32))
        y_pred = K.minimum(y_pred, tf.constant(max_outcome, tf.float32))
    
        return _mae(y_true, y_pred)
    
    return mae
    
def class_filter_rmse_metric(max_outcome, class_filter, offset_factor = 1):
    rmse_calc = rmse_metric(max_outcome)
    
    def class_filter_rmse(y_true, y_pred):
        y_pred = y_pred * offset_factor
        
        idx = tf.where(tf.math.not_equal(tf.cast(class_filter, K.floatx()), y_true))
 
        true = tf.gather(y_true, idx)
        pred = tf.gather(y_pred, idx)

        idx_size = tf.size(idx)

        rmse_val = tf.cond(idx_size == 0,
                    lambda: tf.constant(0, tf.float32),
                    lambda: rmse_calc(true, pred))
        
        return rmse_val
        
    class_filter_rmse.__name__ = 'filter_{}_rmse'.format(class_filter)
    
    return class_filter_rmse
    
def _mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def _rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))