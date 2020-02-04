import tensorflow as tf

import dataset.dataset_ops as ops

class base_dataset():
    def __init__(self, config):
        self.config = config

    def _create_dataset(self, x, y, file_location, update_labels = False):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = ops.load_images(dataset, file_location, update_labels = update_labels)
    
        return dataset

    def _create_validation_split(self, dataset, split_size = 50):
        val_dataset = dataset.take(split_size)
        dataset = dataset.skip(split_size)

        return dataset, val_dataset

    def _prepare_for_training(self, dataset, image_width, image_height, batch_size = 25, cache = True, update_labels = False, augment = True):
        dataset = ops.cache_dataset(dataset, cache)
        dataset = ops.shuffle_and_repeat_dataset(dataset)

        if(augment):
            dataset = ops.randomly_augment_images(dataset, update_labels = update_labels)

        dataset = ops.resize_images(dataset, image_width, image_height, update_labels = update_labels)
    
        dataset = ops.batch_and_prefetch_dataset(dataset, batch_size)
        
        return dataset