import tensorflow as tf

import dataset.ops.dataset_ops as ds_ops

# TODO: Don't use config in constructor, but individual fields passed on from config
class base_dataset():
    def __init__(self, config):
        self.config = config

    def _create_dataset(self, x, y, file_location, update_labels = False, imagenet = False):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size = 20000, seed = 63)
        dataset = ds_ops.load_images(dataset, file_location, update_labels = update_labels, imagenet = imagenet)
    
        return dataset

    def _create_validation_split(self, dataset, split_size = 50):
        val_dataset = dataset.take(split_size)
        dataset = dataset.skip(split_size)

        return dataset, val_dataset

    def _cache_shuffle_repeat_dataset(self, dataset, cache = True, buffer_size = 200):
        dataset = ds_ops.cache_dataset(dataset, cache)
        dataset = ds_ops.shuffle_and_repeat_dataset(dataset, buffer_size = buffer_size)

        return dataset

    def _prepare_for_training(self, dataset, img_height, img_width, batch_size = 64, update_labels = False, augment = True, pad_resize = True):
        if(augment):
            dataset = ds_ops.randomly_augment_images(dataset, update_labels = update_labels)

        dataset = ds_ops.resize_images(dataset, img_height, img_width, update_labels = update_labels, pad_resize = pad_resize)
        dataset = ds_ops.batch_and_prefetch_dataset(dataset, batch_size)
        
        return dataset