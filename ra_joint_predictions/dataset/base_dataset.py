import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import dataset.ops.dataset_ops as ds_ops
import dataset.ops.image_ops as img_ops
import dataset.ops.joint_ops as joint_ops
import model.joint_damage_model as joint_damage_model

from utils.class_weight_utils import calc_adapted_class_weights, calc_relative_class_weights, calc_ln_class_weights

hand_wrist_keys = ['w1', 'w2', 'w3']

# TODO: Don't use config in constructor, but individual fields passed on from config
class base_dataset():
    def __init__(self, config):
        self.config = config
        self.is_wrist = False
        self.is_chest = False
        self.apply_clahe = False

    def _create_dataset(self, x, y, file_location, update_labels = False, imagenet = False):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size = 20000, seed = 63)
        dataset = ds_ops.load_images(dataset, file_location, update_labels = update_labels, imagenet = imagenet)
    
        return dataset

    def _create_validation_split(self, dataset, split_size = 50):
        val_dataset = dataset.take(split_size)
        dataset = dataset.skip(split_size)

        return dataset, val_dataset

    def _cache_shuffle_repeat_dataset(self, dataset, cache = True, buffer_size = 200, do_shuffle = True):
        dataset = ds_ops.cache_dataset(dataset, cache)

        if do_shuffle:
            dataset = ds_ops.shuffle_and_repeat_dataset(dataset, buffer_size = buffer_size)

        return dataset

    def _prepare_for_training(self, dataset, img_height, img_width, batch_size = 64, update_labels = False, augment = True, pad_resize = True):
        if augment:
            if self.is_wrist:
                augments = [
                    {
                        'augment': img_ops.random_brightness_and_contrast
                    },
                    {
                        'augment': img_ops.random_crop
                    },
                    {
                        'augment': img_ops.random_gaussian_noise,
                        'p': 0.2
                    },
                    {
                        'augment': img_ops.random_rotation
                    }]
            elif self.is_chest:
                augments = [
                    {
                        'augment': img_ops.random_flip,
                        'p': 1,
                        'params': {'flip_up_down': False}
                    },
                    {
                        'augment': img_ops.random_brightness_and_contrast
                    },
                    {
                        'augment': img_ops.random_crop
                    },
                    {
                        'augment': img_ops.random_gaussian_noise,
                        'p': 0.2
                    },
                    {
                        'augment': img_ops.random_rotation
                    }]
            else:
                augments = ds_ops.default_augments
        else:
            augments = []
        
        dataset = ds_ops.augment_and_resize_images(dataset, img_height, img_width, update_labels = update_labels, pad_resize = pad_resize, augments = augments)
        dataset = ds_ops.batch_and_prefetch_dataset(dataset, batch_size)
        
        return dataset
    
class joint_dataset(base_dataset):
    def __init__(self, config, cache_postfix = '', imagenet = False, pad_resize = False, joint_extractor = None):
        super().__init__(config)
        self.imagenet = imagenet
        self.cache = config.cache_loc + 'dream/' + cache_postfix
        self.joint_height = config.joint_img_height
        self.joint_width = config.joint_img_width
        self.pad_resize = pad_resize
        self.joint_extractor = joint_extractor

    def _create_joint_dataset(self, file_info, joint_coords, outcomes, wrist = False):
        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords, outcomes))
        if wrist:
            dataset = joint_ops.load_wrists(dataset, self.image_dir, imagenet = self.imagenet)
        else:
            dataset = joint_ops.load_joints(dataset, self.image_dir, imagenet = self.imagenet, joint_extractor = self.joint_extractor, apply_clahe = self.apply_clahe)
        
        return dataset

    def _create_non_split_joint_dataset(self, file_info, coords, outcomes, cache = True, wrist = False, augment = True, buffer_size = 200):
        dataset = self._create_joint_dataset(file_info, coords, outcomes, wrist)

        dataset = self._cache_shuffle_repeat_dataset(dataset, cache = cache, buffer_size = buffer_size)
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize, augment = augment)
        
        return dataset

    def _create_intermediate_joints_df(self, joints_source, joint_keys):
        joints_df = pd.read_csv(joints_source)

        mapped_joints = []
        for _, row in joints_df.iterrows():
            image_name = row['image_name']
  
            for key in joint_keys:
                mapped_joint = {
                    'image_name': image_name,
                    'key': key,
                    'flip': row['flip'],
                    'file_type': row['file_type'],
                    'coord_x': row[key + '_x'],
                    'coord_y': row[key + '_y']
                }

                mapped_joints.append(mapped_joint)

        return pd.DataFrame(mapped_joints, index = np.arange(len(mapped_joints)))
        
    def _create_intermediate_wrists_df(self, joints_source, joint_keys):
        joints_df = pd.read_csv(joints_source)

        mapped_joints = []
        for _, row in joints_df.iterrows():
            image_name = row['image_name']
            mapped_joint = {
                'image_name': image_name,
                'key': "wrist",
                'flip': row['flip'],
                'file_type': row['file_type'],
            }

            for key in joint_keys:
                mapped_joint[key + "_x"] = row[key + '_x']
                mapped_joint[key + "_y"] = row[key + '_y']
                
            mapped_joints.append(mapped_joint)

        return pd.DataFrame(mapped_joints, index = np.arange(len(mapped_joints)))

class dream_dataset(joint_dataset):
    def __init__(self, config, cache_postfix = '', model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None, apply_clahe = False):
        super().__init__(config, 'dream/' + cache_postfix, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location
        self.batch_size = config.batch_size
        self.model_type = model_type
        self.cache = self.cache + '_' + model_type
        self.split_type = split_type
        self.maj_ratio = 0.25
        self.apply_clahe = apply_clahe

    def _create_dream_datasets(self, outcomes_source, joints_source, outcome_mapping, parts, outcome_columns, no_classes, wrist = False):
        outcome_joint_df = self._create_outcome_joint_dataframe(outcomes_source, joints_source, outcome_mapping, parts, wrist = wrist)
        outcome_joint_df = outcome_joint_df.dropna(subset = outcome_columns)

        dataset = self._create_dream_dataset(outcome_joint_df, outcome_columns, no_classes, cache = self.cache, wrist = wrist)
        
        return dataset

    def _create_outcome_joint_dataframe(self, outcomes_source, joints_source, outcome_mapping, parts, wrist = False):
        outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, outcome_mapping, parts)
        if wrist:
            joints_df = self._create_intermediate_wrists_df(joints_source, hand_wrist_keys)
        else:
            joints_df = self._create_intermediate_joints_df(joints_source, outcome_mapping.keys())

        return outcomes_df.merge(joints_df, on = ['image_name', 'key'])

    def _create_intermediate_outcomes_df(self, outcomes_source, outcome_mapping, parts):
        outcomes_df = pd.read_csv(outcomes_source)

        outcome_joints = []
        for _, row in outcomes_df.iterrows():
            patient_id = row['Patient_ID']

            for part in parts:
                image_name = patient_id + '-' + part

                for idx, key in enumerate(outcome_mapping.keys()):
                    joint_mapping = outcome_mapping[key]

                    outcome_joint = {
                        'image_name': image_name,
                        'key': key
                    }
                    
                    mapped_narrowing_keys = [key_val.format(part = part) for key_val in joint_mapping[0]]
                    for idx, mapped_key in enumerate(mapped_narrowing_keys):
                        outcome_joint[f'narrowing_{idx}'] = row[mapped_key]

                    mapped_erosion_keys = [key_val.format(part = part) for key_val in joint_mapping[1]]
                    for idx, mapped_key in enumerate(mapped_erosion_keys):
                        outcome_joint[f'erosion_{idx}'] = row[mapped_key]

                    outcome_joints.append(outcome_joint)

        return pd.DataFrame(outcome_joints, index = np.arange(len(outcome_joints)))    

    def _create_dream_dataset(self, outcome_joint_df, outcome_columns, no_classes, augment = True, cache = True, wrist = False, is_train = True):
        outcome_joint_df = outcome_joint_df.sample(frac = 1).reset_index(drop = True)

        file_info = outcome_joint_df[['image_name', 'file_type', 'flip', 'key']].values

        outcomes = outcome_joint_df[outcome_columns]
        if is_train:
            self._init_model_outcomes_bias(outcomes, no_classes)

        tf_dummy_outcomes, tf_outcomes = self._get_outcomes(outcomes, no_classes)
        
        if wrist:
            coords = outcome_joint_df[['w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']].values
        else:
            coords = outcome_joint_df[['coord_x', 'coord_y']].values

        if is_train:
            maj_idx = self._find_maj_indices(outcomes)

            return self._create_interleaved_joint_datasets(file_info, coords, maj_idx, outcomes = tf_outcomes, dummy_outcomes = tf_dummy_outcomes, wrist = wrist)
        else:
            return self._create_non_split_joint_dataset(file_info, coords, tf_outcomes, wrist = wrist, augment = False)

    def _find_maj_indices(self, outcomes):
        return outcomes == 0
    
    def _get_outcomes(self, outcomes, no_classes):
        tf_dummy_outcomes = None
        tf_outcomes = None
        
        if self.model_type == joint_damage_model.MODEL_TYPE_CLASSIFICATION:
            tf_dummy_outcomes = self._dummy_encode_outcomes(outcomes, no_classes)
        elif self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
            tf_outcomes = outcomes.to_numpy(dtype = np.float64)
            self.outcomes = tf_outcomes
        elif self.model_type == joint_damage_model.MODEL_TYPE_COMBINED:
            tf_dummy_outcomes = self._dummy_encode_outcomes(outcomes, no_classes)
            tf_outcomes = outcomes.to_numpy()
            
        return tf_dummy_outcomes, tf_outcomes

    def _create_interleaved_joint_datasets(self, file_info, joint_coords, maj_idx, outcomes = None, dummy_outcomes = None, wrist = False):
        if self.split_type == 'balanced':
            dataset = self._create_fully_balanced_dataset(file_info, joint_coords, maj_idx, outcomes = outcomes, dummy_outcomes = dummy_outcomes, wrist = wrist)
        elif self.split_type == 'minority-balanced':
            dataset = self._create_fully_balanced_dataset(file_info, joint_coords, maj_idx, outcomes = outcomes, dummy_outcomes = dummy_outcomes, wrist = wrist, only_minority = True)
        elif self.split_type == 'minority':
            dataset = self._create_minority_dataset(file_info, joint_coords, maj_idx, outcomes = outcomes, dummy_outcomes = dummy_outcomes, wrist = wrist)
        elif self.split_type == 'none':
            # Create 2 datasets, one with the majority class, one with the other classes
            dataset = self._create_joint_dataset(file_info, joint_coords, outcomes, wrist = wrist)
            
            # Cache the partial datasets, shuffle the datasets with buffersize that ensures minority samples are all shuffled
            dataset = self._cache_shuffle_repeat_dataset(dataset, self.cache, buffer_size = file_info.shape[0])
        else:
            dataset = self._create_50_50_ds(file_info, joint_coords, maj_idx, outcomes = outcomes, dummy_outcomes = dummy_outcomes, wrist = wrist)
        
        # Prepare for training
        dataset = self._prepare_for_training(dataset, self.joint_height, self.joint_width, batch_size = self.config.batch_size, pad_resize = self.pad_resize)

        return dataset
    
    def _create_50_50_ds(self, file_info, joint_coords, maj_idx, outcomes = None, dummy_outcomes = None, wrist = False):
        min_idx = np.logical_not(maj_idx)

        # Tranform boolean mask into indices
        maj_idx = np.where(maj_idx)[0]
        min_idx = np.where(min_idx)[0]

        if self.model_type == 'C':
            maj_outcomes = dummy_outcomes[maj_idx, :]
            min_outcomes = dummy_outcomes[min_idx, :]
        elif self.model_type == 'R':
            maj_outcomes = outcomes[maj_idx]
            min_outcomes = outcomes[min_idx]
        elif self.model_type == 'RC':
            maj_outcomes = (outcomes[maj_idx], dummy_outcomes[maj_idx, :])
            min_outcomes = (outcomes[min_idx], dummy_outcomes[min_idx, :])

        # Create 2 datasets, one with the majority class, one with the other classes
        maj_ds = self._create_joint_dataset(file_info[maj_idx, :], joint_coords[maj_idx], maj_outcomes, wrist = wrist)
        min_ds = self._create_joint_dataset(file_info[min_idx, :], joint_coords[min_idx], min_outcomes, wrist = wrist)

        # Cache the partial datasets, shuffle the datasets with buffersize that ensures minority samples are all shuffled
        maj_ds = self._cache_shuffle_repeat_dataset(maj_ds, self.cache + '_maj', buffer_size = maj_idx.shape[0])
        min_ds = self._cache_shuffle_repeat_dataset(min_ds, self.cache + '_min', buffer_size = min_idx.shape[0])

        # Interleave datasets 50/50 - for each majority sample (class 0), it adds one none majority sample (not class 0)
        dataset = tf.data.experimental.sample_from_datasets((maj_ds, min_ds), [self.maj_ratio, 1 - self.maj_ratio])
        
        return dataset
    
    def _create_fully_balanced_dataset(self, file_info, joint_coords, maj_idx, outcomes = None, dummy_outcomes = None, wrist = False, only_minority = False):
        if self.model_type == 'C':
            outcomes = dummy_outcomes

        idx_groups = self._get_idx_groups(outcomes)

        if only_minority:
            # res = [0.04, 0.23, 0.23, 0.23, 0.23]
            
            idx_groups = idx_groups[1:]
        else:
            res = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        datasets = []
        
        for n, idx_group in enumerate(idx_groups):
            idx = np.where(idx_group)[0]
            
            ds_outcomes = outcomes[idx, :]
            ds = self._create_joint_dataset(file_info[idx, :], joint_coords[idx], ds_outcomes, wrist = wrist)
            ds = self._cache_shuffle_repeat_dataset(ds, self.cache + str(n), buffer_size = idx.shape[0])
            
            datasets.append(ds)
            
        dataset = tf.data.experimental.sample_from_datasets(datasets) 
        
        return dataset
    
    def _create_minority_dataset(self, file_info, joint_coords, maj_idx, outcomes = None, dummy_outcomes = None, wrist = False):
        min_idx = np.logical_not(maj_idx)
        min_idx = np.where(min_idx)[0]
        
        if self.model_type == 'C':
            min_outcomes = dummy_outcomes[min_idx, :]
        elif self.model_type == 'R':
            min_outcomes = outcomes[min_idx]
        elif self.model_type == 'RC':
            min_outcomes = (outcomes[min_idx], dummy_outcomes[min_idx, :])
        
        min_ds = self._create_joint_dataset(file_info[min_idx, :], joint_coords[min_idx], min_outcomes, wrist = wrist)
        min_ds = self._cache_shuffle_repeat_dataset(min_ds, self.cache + '_min_ds', buffer_size = min_idx.shape[0])

        return min_ds
    
    def _init_model_outcomes_bias(self, outcomes, no_classes):
        self.class_weights = calc_relative_class_weights(outcomes, no_classes)

    def _dummy_encode_outcomes(self, outcomes, no_classes):
        D = outcomes.shape[1]

        one_hot_encoder = OneHotEncoder(sparse = False, categories = [np.arange(no_classes)] * D)
        column_transformer = ColumnTransformer([('one_hot_encoder', one_hot_encoder, np.arange(D))], remainder = 'passthrough')

        return column_transformer.fit_transform(outcomes.to_numpy()).astype(dtype = np.float64)
