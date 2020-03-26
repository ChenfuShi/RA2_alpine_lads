import tensorflow as tf

def default_joint_extractor(joint_scale = 5):
    def _default_joint_extractor(img_shape, joint_key):
        box_height = img_shape[0] / joint_scale
        box_width = img_shape[1] / joint_scale

        return box_height, box_width

    return _default_joint_extractor

def feet_joint_extractor(main_joint_scale = 6, ntp_joint_scale = 3):
    def _feet_joint_extractor(img_shape, joint_key):
        if joint_key == 'mtp' or joint_key == 'mtp_1':
            joint_scale = ntp_joint_scale
        
        joint_scale = tf.cast(joint_scale, tf.float64)
    
        box_height = img_shape[0] / joint_scale
        box_width = img_shape[1] / joint_scale

        return box_height, box_width

    return _feet_joint_extractor