import os
from utils.config import Config
import PIL.Image
import PIL.ImageOps
import cv2
import numpy as np
import time

start = time.time()

config = Config('./utils/stadlerm_config.json')

train_location = config.train_location
fixed_dir = config.fixed_dir

filenames = os.listdir(train_location)

for idx, filename in enumerate(filenames):
    if filename.endswith(".jpg"):
        file_path = os.path.join(train_location, filename)
        
        img = PIL.Image.open(file_path)
        img_mode = img.mode
    
        # Apply details from EXIF
        img = PIL.ImageOps.exif_transpose(img)
        
        img = np.asarray(img)
        if(img_mode == 'RGB'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        #Write the adjusted images to the fixed dir
        cv2.imwrite(os.path.join(train_location, fixed_dir, filename), img)
        
        #img.save(os.path.join(train_location, fixed_dir, filename))
        
end = time.time()
print('Runtime: ', end - start)