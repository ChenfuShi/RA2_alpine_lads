import tensorflow as tf

import dataset.ops.image_ops as img_ops

from dataset.joints.joint_exractor import default_joint_extractor

AUTOTUNE = tf.data.experimental.AUTOTUNE

round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

feet_joints = ['mtp', 'mtp_1', 'mtp_2', 'mtp_3', 'mtp_4', 'mtp_5']

def load_joints(dataset, directory, imagenet = False, joint_extractor = default_joint_extractor()):
    def __load_joints(file_info, y, z):
        joint_key = file_info[3]

        x_coord = y[0]
        y_coord = y[1]

        full_img, _ = img_ops.load_image(file_info, [], directory, imagenet = imagenet)

        joint_img = _extract_joint_from_image(full_img, joint_key, x_coord, y_coord, joint_extractor)

        return joint_img, z

    return dataset.map(__load_joints, num_parallel_calls=AUTOTUNE)
    
def load_wrists(dataset, directory, imagenet = False):
    def __load_wrists(file_info, y, z):
        w1_x = y[0]
        w2_x = y[2]
        w3_x = y[4]
        w1_y = y[1]
        w2_y = y[3]
        w3_y = y[5]

        full_img, _ = img_ops.load_image(file_info, [], directory, imagenet = imagenet)

        joint_img = _extract_wrist_from_image(full_img, w1_x, w2_x, w3_x, w1_y, w2_y, w3_y)

        return joint_img, z

    return dataset.map(__load_wrists, num_parallel_calls=AUTOTUNE)

def _extract_joint_from_image(img, joint_key, x, y, joint_extractor):
    img_shape = tf.cast(tf.shape(img), tf.float64)
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    
    box_height, box_width = joint_extractor(img_shape, joint_key)

    # get top left corner of image
    x_box = x - (box_width / 2)
    y_box = y - (box_height / 2)

    # make sure top left is within image
    x_box = tf.math.maximum(x_box, 10)
    y_box = tf.math.maximum(y_box, 10)

    # make sure top left is within image from the other two sides
    x_box = tf.math.minimum(x_box, img_shape[1] - 10)
    y_box = tf.math.minimum(y_box, img_shape[0] - 10)

    # make sure the resulting box is within the image, if it's too large new height and width to be within image
    if y_box + box_height > img_shape[0]:
        box_height = img_shape[0] - y_box

    if x_box + box_width > img_shape[1]:
        box_width = img_shape[1] - x_box
        
    img = tf.image.crop_to_bounding_box(img, round_to_int(y_box), round_to_int(x_box), round_to_int(box_height), round_to_int(box_width))

    return img


def _extract_wrist_from_image(img, w1_x, w2_x, w3_x, w1_y, w2_y, w3_y):
    img_shape = tf.cast(tf.shape(img), tf.float64)
    w1_x, w2_x, w3_x, w1_y, w2_y, w3_y = [tf.cast(x, tf.float64) for x in [w1_x, w2_x, w3_x, w1_y, w2_y, w3_y]]

    extra_pad_height = img_shape[0] / 15
    extra_pad_width = img_shape[1] / 15

    # identify left top most points
    x_box = tf.reduce_min(tf.stack([w1_x, w2_x, w3_x]),0) - extra_pad_width
    y_box = tf.reduce_min(tf.stack([w1_y, w2_y, w3_y]),0) - extra_pad_height

    # make sure they are within the image
    x_box = tf.math.maximum(x_box, 10)
    y_box = tf.math.maximum(y_box, 10)
    x_box = tf.math.minimum(x_box, img_shape[1] - 11)
    y_box = tf.math.minimum(y_box, img_shape[0] - 11)

    # get the bottom right most point
    x_box_max = tf.reduce_max(tf.stack([w1_x, w2_x, w3_x]),0) + extra_pad_width
    y_box_max = tf.reduce_max(tf.stack([w1_y, w2_y, w3_y]),0) + extra_pad_height

    # make sure they are within the image
    x_box_max = tf.math.maximum(x_box_max, 11)
    y_box_max = tf.math.maximum(y_box_max, 11)
    x_box_max = tf.math.minimum(x_box_max, img_shape[1] - 10)
    y_box_max = tf.math.minimum(y_box_max, img_shape[0] - 10)

    # calculate the resulting height and width
    box_height = y_box_max - y_box
    box_width = x_box_max - x_box

    # # make sure they are at least a certain size
    box_height = tf.math.maximum(box_height, tf.constant(50,dtype=tf.float64))
    box_width = tf.math.maximum(box_width, tf.constant(50,dtype=tf.float64))

    # hope
    img = tf.image.crop_to_bounding_box(img, round_to_int(y_box), round_to_int(x_box), round_to_int(box_height), round_to_int(box_width))

    return img