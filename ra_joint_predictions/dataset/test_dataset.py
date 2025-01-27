import numpy as np

import pandas as pd
import tensorflow as tf

import dataset.ops.image_ops as img_ops
import dataset.ops.dataset_ops as ds_ops
import dataset.joint_dataset as joint_dataset
import dataset.ops.joint_ops as joint_ops
import model.joint_damage_model as joint_damage_model

from dataset.base_dataset import dream_dataset
from dataset.joints.joint_extractor import default_joint_extractor

AUTOTUNE = tf.data.experimental.AUTOTUNE

class joint_test_dataset(dream_dataset):
    def __init__(self, config, img_dir, model_type = 'R', pad_resize = False, imagenet = False, joint_extractor = default_joint_extractor(), apply_clahe = False, repeat = True):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.img_dir = img_dir
        self.pad_resize = pad_resize
        self.apply_clahe = apply_clahe
        self.repeat = repeat
        self.divide_erosion_by_2 = False
        
    def get_hands_joint_test_dataset(self, joints_source = './data/predictions/hand_joint_data_test.csv', outcomes_source = None, erosion_flag = None):
        if erosion_flag is False:
            params = joint_dataset.hands_narrowing_params
        elif erosion_flag is True:
            params = joint_dataset.hands_erosion_params
        else: 
            params = None
        
        return self._create_joint_dataset(joints_source, joint_dataset.hand_outcome_mapping, outcomes_source, params)

    def get_wrists_joint_test_dataset(self, joints_source = './data/predictions/hand_joint_data_test.csv', outcomes_source = None, erosion_flag = None):
        if erosion_flag is False:
            params = joint_dataset.wrists_narrowing_params
        elif erosion_flag is True:
            params = joint_dataset.wrists_erosion_params
        else: 
            params = None
        
        return self._create_joint_dataset(joints_source, joint_dataset.wrist_outcome_mapping, outcomes_source, params, load_wrists = True)
        
    def get_feet_joint_test_dataset(self, joints_source = './data/predictions/feet_joint_data_test.csv', outcomes_source = None, erosion_flag = None, divide_by_two = False):
        if erosion_flag is False:
            params = joint_dataset.feet_narrowing_params
        elif erosion_flag is True:
            params = joint_dataset.feet_erosion_params
            self.divide_erosion_by_2 = divide_by_two
        else: 
            params = None
        
        return self._create_joint_dataset(joints_source, joint_dataset.foot_outcome_mapping, outcomes_source, params)

    def _create_joint_dataset(self, joints_source, outcome_mapping, outcomes_source, params, load_wrists = False):
        df = self._create_df(joints_source, outcome_mapping, outcomes_source, params, load_wrists = load_wrists)
        
        if outcomes_source is not None:
            dataset, no_samples = self._create_dataset(df, params, load_wrists)
        else:
            dataset, no_samples = self._create_dataset(df, None, load_wrists)

        return dataset, no_samples

    def _create_df(self, joints_source, outcome_mapping, outcomes_source, params, load_wrists = False):
        if load_wrists:
            joints_df = self._create_intermediate_wrists_df(joints_source, joint_dataset.hand_wrist_keys)
        else:
            joints_df = self._create_intermediate_joints_df(joints_source, outcome_mapping.keys())

        # If outcomes are provided, add them to the dataframe
        if outcomes_source:
            outcome_df = self._create_intermediate_outcomes_df(outcomes_source, outcome_mapping, params['parts'])
            outcome_df = outcome_df.dropna(subset = params['outcomes'])

            joints_df = joints_df.merge(outcome_df, on = ['image_name', 'key'])
            
        return joints_df

    def _create_dataset(self, df, params, load_wrists):
        if params:
            df = df.sample(frac = 1, random_state = 1337).reset_index(drop = True)
        
        file_info = df[['image_name', 'file_type', 'flip', 'key']].to_numpy()

        if load_wrists:
            joint_coords = df[['w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']].to_numpy()
        else:
            joint_coords = df[['coord_x', 'coord_y']].to_numpy()

        outcomes = self._get_outcomes(df, params)

        self.outcomes = outcomes
        
        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords, outcomes))

        if load_wrists:
            dataset = self._load_wrists_without_outcomes(dataset)
        else:
            dataset = self._load_joints_without_outcomes(dataset, apply_clahe = self.apply_clahe)

        dataset = self._resize_images_without_outcomes(dataset)
        
        if params:
            dataset = self._remove_file_info(dataset)

            if load_wrists:
                dataset = self._split_outcomes(dataset, params['no_classes'])
           
            #if self.model_type == 'DT':
                #dataset = ds_ops.shuffle_and_repeat_dataset(dataset, buffer_size = 2000)
        
            if self.repeat:
                dataset = dataset.cache()
                dataset = dataset.repeat()
                
            dataset = dataset.batch(self.config.batch_size)
        else:
            dataset = self._remove_outcome(dataset)

        return dataset.prefetch(buffer_size = AUTOTUNE), file_info.shape[0]

    def _get_outcomes(self, df, params):
        if params:
            outcomes = df[params['outcomes']]
            
            tf_dummy_outcomes = None
            tf_outcomes = None

            if self.model_type == joint_damage_model.MODEL_TYPE_CLASSIFICATION:
                tf_dummy_outcomes = self._dummy_encode_outcomes(outcomes, params['no_classes'])

                outcomes = tf_dummy_outcomes
            elif self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
                tf_outcomes = outcomes.to_numpy()
                
                if self.divide_erosion_by_2:
                    tf_outcomes = tf_outcomes / 2

                outcomes = tf_outcomes
            elif self.model_type == joint_damage_model.MODEL_TYPE_COMBINED:
                tf_dummy_outcomes = self._dummy_encode_outcomes(outcomes, params['no_classes'])
                tf_outcomes = outcomes.to_numpy()

                outcomes = (tf_outcomes, tf_dummy_outcomes)
            elif self.model_type == 'DT':
                maj_idx = outcomes == 0

                # Set majority samples to 0
                tf_outcomes = np.ones(df.shape[0])
                tf_outcomes[np.where(maj_idx)[0]] = 0
                
                # tf_outcomes = outcomes.to_numpy()
                
                outcomes = tf_outcomes
        else:
            outcomes = np.zeros(df.shape[0])

        return outcomes
        
    def _load_joints_without_outcomes(self, dataset, apply_clahe = False):
        def __load_joints(file_info, coords, y):
            x_coord = coords[0]
            y_coord = coords[1]

            full_img, _ = img_ops.load_image(file_info, [], self.img_dir, imagenet = self.imagenet)

            joint_img = joint_ops._extract_joint_from_image(full_img, file_info[3], x_coord, y_coord, joint_extractor = self.joint_extractor)

            if apply_clahe:
                joint_img = img_ops.clahe_img(joint_img)

            return file_info, joint_img, y

        return dataset.map(__load_joints, num_parallel_calls=AUTOTUNE)

    def _load_wrists_without_outcomes(self, dataset):
        def __load_wrists(file_info, coords, y):
            w1_x = coords[0]
            w2_x = coords[2]
            w3_x = coords[4]
            w1_y = coords[1]
            w2_y = coords[3]
            w3_y = coords[5]

            full_img, _ = img_ops.load_image(file_info, [], self.img_dir, imagenet = self.imagenet)

            joint_img = joint_ops._extract_wrist_from_image(full_img, w1_x, w2_x, w3_x, w1_y, w2_y, w3_y)

            return file_info, joint_img, y

        return dataset.map(__load_wrists, num_parallel_calls=AUTOTUNE)

    def _resize_images_without_outcomes(self, dataset):
        def __resize(file_info, img, y):
            img, _ =  img_ops.resize_image(img, [], self.config.joint_img_height, self.config.joint_img_width, pad_resize = self.pad_resize, update_labels = False)

            return file_info, img, y

        return dataset.map(__resize, num_parallel_calls = AUTOTUNE)

    def _remove_file_info(self, dataset):
        def _remove_file_info(file_info, img, y):
            return img, y

        return dataset.map(_remove_file_info, num_parallel_calls = AUTOTUNE)

    def _split_outcomes(self, dataset, no_classes):
        if self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
            no_classes = 1
        
        def __split_outcomes(x, y):
            split_y = tf.split(y, 6, -1)

            return x, (split_y[0], split_y[1], split_y[2], split_y[3], split_y[4], split_y[5])

        return dataset.map(__split_outcomes, num_parallel_calls=AUTOTUNE)

    def _remove_outcome(self, dataset):
        def __remove_outcome(file_info, img, y):
            return file_info, img

        return dataset.map(__remove_outcome, num_parallel_calls = AUTOTUNE)

class combined_test_dataset(joint_test_dataset, joint_dataset.combined_joint_dataset):
    def __init__(self, config, img_dir, model_type = 'R', pad_resize = False, joint_extractor = None):
        super().__init__(config, img_dir, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor)

    def get_combined_joint_test_dataset(self, hand_joints_source = './data/predictions/hand_joint_data_test_v2.csv', feet_joints_source = './data/predictions/feet_joint_data_test_v2.csv', outcomes_source = None, erosion_flag = False):
        if not erosion_flag:
            params = joint_dataset.hands_narrowing_params
        else:
            params = joint_dataset.hands_erosion_params

        self.outcome_columns = params['outcomes']

        combined_joints_df = self._create_combined_df(outcomes_source, hand_joints_source, feet_joints_source)

        return self._create_dataset(combined_joints_df, params, False)
