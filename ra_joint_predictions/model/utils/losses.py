import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

def focal_loss(gamma = 2., alpha = 0.25):
    def _focal_loss_fixed(y_true, y_pred):
        return tfa.losses.focal_loss.sigmoid_focal_crossentropy(y_true, y_pred, gamma = gamma, alpha = alpha)
    
    return _focal_loss_fixed

def softmax_focal_loss(alpha, gamma = 2.):
    alpha = K.constant(alpha)
    gamma = K.constant(gamma)
    
    def _softmax_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Find the correct class weights
        alpha_factor = tf.reduce_sum(y_true * alpha, axis = 1)
        
        # Find the probability of the true class
        gamma_factor = K.pow(1 - tf.reduce_sum(y_pred * y_true, axis = 1), gamma)
        
        # Calculate the CE loss
        categorical_crossentropy = K.categorical_crossentropy(y_true, y_pred)
        
        return categorical_crossentropy * gamma_factor * alpha_factor

    return _softmax_focal_loss_fixed