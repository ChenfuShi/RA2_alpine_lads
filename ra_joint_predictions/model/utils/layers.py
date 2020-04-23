import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.ops import math_ops

class ReLUOutput(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLUOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.threshold = self.add_weight(name='threshold',
                                      initializer = tf.keras.initializers.Constant(0.2),
                                      trainable = False)
        super(ReLUOutput, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        threshold = math_ops.cast(self.kernel, x.dtype)
        
        return K.relu(x, threshold = K.get_value(threshold))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return super(ReLUOutput, self).get_config()