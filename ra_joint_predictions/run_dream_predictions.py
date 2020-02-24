import os
import tensorflow as tf
import tensorflow_addons as tfa

from utils.config import Config
from dream import execute_dream_predictions
from dream.preprocessing import image_preprocessing

# Change to dir
os.chdir('/usr/local/bin/ra_joint_predictions/')

print('----------------')
print('Running version:', os.environ['CURR_VERSION'])
print('----------------')
print('Start init.py')
print('Check TF Versions:')
print('TF: ', tf.__version__)
print('TF Addons: ', tfa.__version__) 
print('GPU available: ', tf.test.is_gpu_available())
print('----------------')

config = Config('./utils/docker-config.json')

# Preprocess the images
image_preprocessing(config)
# Run predictions
execute_dream_predictions()
