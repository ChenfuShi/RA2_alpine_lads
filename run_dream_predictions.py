import os
import tensorflow as tf
import tensorflow_addons as tfa

from dream import execute_dream_predictions

# Change to dir
os.chdir('/usr/local/bin/RA2_alpine_lads/')

print('----------------')
print('Start init.py')
print('Check TF Versions:')
print('TF: ', tf.__version__)
print('TF Addons: ', tfa.__version__) 
print('GPU available: ', tf.test.is_gpu_available())
print('----------------')

# Entry point for dream predictions
execute_dream_predictions()