import os
import time

from utils.image_preprocessor import preprocess_images

def image_preprocessing(config):
    print('----------------')
    print('Preprocessing images')

    start = time.time()

    preprocess_images(config.train_location, os.path.join(config.train_location, config.fixed_dir))
    preprocess_images(config.test_location, os.path.join(config.test_location, config.fixed_dir))

    end = time.time()
    print('Preprocessed images in : ', end - start)
    print('----------------')