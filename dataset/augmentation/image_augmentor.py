import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def randomly_augment_dataset(dataset, augments):
    for aug in augments:
        dataset = _apply_random_augment(dataset, aug)
        
    # After augmentations, scale values back to lie between 0 & 1
    
    dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=AUTOTUNE)

    return dataset

def _apply_random_augment(dataset, aug, cutoff = 0.6):
    def __apply_random_augment(x, y):
        return tf.cond(tf.random.uniform([], 0, 1) > cutoff, lambda: aug(x, y), lambda: (x, y))
    
    return dataset.map(__apply_random_augment, num_parallel_calls=AUTOTUNE)