import tensorflow as tf

import dataset.ops.image_ops as img_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

def load_joints(dataset, directory):
    def __load_joints(file_info, y, z):
        joint_key = file_info[3]

        x_coord = y[0]
        y_coord = y[1]

        full_img, _ = img_ops.load_image(file_info, [], directory)

        #TODO: Use joint_key to decide on box dimensions

        joint_img = _extract_joint_from_image(full_img, x_coord, y_coord)

        return joint_img, z

    return dataset.map(__load_joints, num_parallel_calls=AUTOTUNE)

def _extract_joint_from_image(img, x, y):
    img_shape = tf.cast(tf.shape(img), tf.float64)
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    box_height = img_shape[0] / 5
    box_width = img_shape[1] / 5

    x_box = x - (box_width / 2)
    y_box = y - (box_height / 2)

    x_box = tf.math.maximum(x_box, 0)
    y_box = tf.math.maximum(y_box, 0)

    if y_box + box_height > img_shape[0]:
        box_height = img_shape[0] - y_box

    if x_box + box_width > img_shape[1]:
        box_width = img_shape[1] - x_box

    img = tf.image.crop_to_bounding_box(img, round_to_int(y_box), round_to_int(x_box), round_to_int(box_height), round_to_int(box_width))

    return img