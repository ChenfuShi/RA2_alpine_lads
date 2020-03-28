import tensorflow as tf

def default_joint_extractor(joint_scale = 5):
    def _default_joint_extractor(img_shape, joint_key):
        box_height = img_shape[0] / joint_scale
        box_width = img_shape[1] / joint_scale

        return box_height, box_width

    return _default_joint_extractor

def feet_joint_extractor(main_joint_scale = 6, mtp_joint_scale = 3):
    def _feet_joint_extractor(img_shape, joint_key):
        if joint_key == 'mtp' or joint_key == 'mtp_1':
            joint_scale = mtp_joint_scale
        else:
            joint_scale = main_joint_scale
        
        joint_scale = tf.cast(joint_scale, tf.float64)
    
        box_height = img_shape[0] / joint_scale
        box_width = img_shape[1] / joint_scale

        return box_height, box_width

    return _feet_joint_extractor

def _build_lookup_table_from_dict(dictionary, default_value):
    # Init a default so it works if the dictionary is empty
    keys = ['default']
    values = [0.]
    
    keys.extend(list(dictionary.keys()))
    values.extend(list(dictionary.values()))
    
    lookup_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys = tf.constant(keys),
            values = tf.constant(values),
        ),
        default_value = tf.constant(default_value),
    )
    
    return lookup_table

def width_based_joint_extractor(joint_scale = 8., height_scale = 0.8, key_joint_scales = {}, key_height_scales = {}):
    joint_scale_table = _build_lookup_table_from_dict(key_joint_scales, joint_scale)
    height_scale_table = _build_lookup_table_from_dict(key_height_scales, height_scale)
    
    def width_based_joint_extractor_fixed(img_shape, joint_key):
        joint_scale = tf.cast(joint_scale_table.lookup(joint_key), img_shape.dtype) 
        height_scale = tf.cast(height_scale_table.lookup(joint_key), img_shape.dtype)
        
        box_width = img_shape[1] / joint_scale
        box_height = box_width * height_scale

        return box_height, box_width
    
    return width_based_joint_extractor_fixed