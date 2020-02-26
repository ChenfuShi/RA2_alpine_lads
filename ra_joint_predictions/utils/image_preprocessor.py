import cv2
import os
import logging

import numpy as np
import PIL.Image
import PIL.ImageOps

from utils.file_utils import is_supported_image_file

supported_file_types = ['jpg', 'png', 'jpeg']

def preprocess_images(image_directory, target_directory):
    processed_images = []
    
    filenames = os.listdir(image_directory)

    for filename in filenames:
        file_parts = filename.split('.')

        if is_supported_image_file(file_parts):
            file_path = os.path.join(image_directory, filename)

            img = PIL.Image.open(file_path)
            img_mode = img.mode
        
            # Apply details from EXIF
            img = PIL.ImageOps.exif_transpose(img)
            
            img = np.asarray(img)
            if(img_mode == 'RGB'):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            #Write the adjusted images to the fixed dir
            written_img = cv2.imwrite(os.path.join(target_directory, filename), img)

            if written_img:
                processed_images.append(filename)
            else:
                logging.warn('Failed to write preprocessed image %s to %s', filename, target_directory)

    return processed_images