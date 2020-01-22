import tensorflow as tf
import PIL.Image
import PIL.ImageOps
import numpy as np

def load_and_orient_image(file):
    # Load the image with PIL
    img = PIL.Image.open(file)
    
    # Apply details from EXIF
    img = PIL.ImageOps.exif_transpose(img)
    
    # Convert image to grayscale
    img = PIL.ImageOps.grayscale(img)
    
    return tf.convert_to_tensor(np.asarray(img))