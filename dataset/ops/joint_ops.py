import tensorflow as tf

import dataset.ops.image_ops as img_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

def load_joints(dataset, directory):
    def __load_joints(file, y):
        file_name = file[0]
        flip = file[1]
        joint_key = file[2]

        x_coord = y[0]
        y_coord = y[1]

        flip_img = flip == 'Y'

        full_img, _ = img_ops.load_image(file_name, y, False, directory, flip_img)

        #TODO: Use joint_key to decide on box dimensions

        joint_img = _extract_joint_from_image(full_img, x_coord, y_coord)

        return joint_img, y[2:]

    return dataset.map(__load_joints, num_parallel_calls=AUTOTUNE)

def _extract_joint_from_image(img, x, y):
    img_shape = tf.cast(tf.shape(img), tf.float64)
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    box_height = img_shape[0] / 5
    box_width = img_shape[1] / 5

    x_box = x - (box_width / 2)
    y_box = y - (box_height / 2)

    img = tf.image.crop_to_bounding_box(img, round_to_int(y_box), round_to_int(x_box), round_to_int(box_height), round_to_int(box_width))

    return img