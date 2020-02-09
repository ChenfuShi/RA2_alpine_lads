import numpy as np
import tensorflow as tf

import dataset.image_ops as img_ops
import dataset.landmark_ops as lm_ops

class joint_detector():
    def __init__(self, image_directory, img_width, img_height):
        self.image_directory = image_directory
        self.img_width = img_width
        self.img_height = img_height      

    def _get_joint_landmark_coords(self, image_name, flip_img, joint_detector):
        image, _ = img_ops.load_image(image_name, [], False, self.image_directory, flip_img)
        # Resize image to size expected by detection network
        landmark_detection_image, _ = img_ops.resize_image(image, [], False, self.img_width, self.img_height)

        # Pass in a batch of 1 single image
        predicted_landmarks = joint_detector.predict(np.array([landmark_detection_image,]))[0]

        # Rescale the found landmarks to the original image size
        predicted_landmarks = lm_ops.upscale_detected_landmarks(predicted_landmarks, landmark_detection_image.shape, image.shape)

        # If this is an image that gets flipped, we preflip the lables here
        if flip_img:
            predicted_landmarks = tf.cast(predicted_landmarks, dtype = tf.float64)
            predicted_landmarks = lm_ops.flip_landmarks(predicted_landmarks, image.shape)
            predicted_landmarks = tf.cast(predicted_landmarks, dtype = tf.float32)

        return predicted_landmarks