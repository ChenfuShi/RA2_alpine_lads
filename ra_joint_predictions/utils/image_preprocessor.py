import cv2
import os

import numpy as np
import PIL.Image
import PIL.ImageOps

supported_file_types = ['jpg', 'png', 'jpeg']

def preprocess_images(image_directory, target_directory):
    filenames = os.listdir(image_directory)
    for filename in filenames:
        file_parts = filename.split('.')

        if(len(file_parts) > 1 and file_parts[1] in supported_file_types):
            file_path = os.path.join(image_directory, filename)

            img = PIL.Image.open(file_path)
            img_mode = img.mode
        
            # Apply details from EXIF
            img = PIL.ImageOps.exif_transpose(img)
            
            img = np.asarray(img)
            if(img_mode == 'RGB'):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            #Write the adjusted images to the fixed dir
            cv2.imwrite(os.path.join(target_directory, filename), img)