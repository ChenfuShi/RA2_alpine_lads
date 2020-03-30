import numpy as np
import pandas as pd
import tensorflow as tf

import prediction.joint_detection as joint_detection

from dataset.base_dataset import dream_dataset
from dataset.ops import dataset_ops as ds_ops
from dataset.ops import image_ops as img_ops
from dataset.ops import joint_ops as js_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

hand_coord_keys = [
    'mcp_x', 'mcp_y', 
    'pip_2_x', 'pip_2_y',
    'pip_3_x', 'pip_3_y',
    'pip_4_x', 'pip_4_y',
    'pip_5_x', 'pip_5_y',
    'mcp_1_x', 'mcp_1_y',
    'mcp_2_x', 'mcp_2_y',
    'mcp_3_x', 'mcp_3_y', 
    'mcp_4_x', 'mcp_4_y',
    'mcp_5_x', 'mcp_5_y',
    'w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']

wrist_coord_keys = [
    'w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y'
]

hand_coord_mapping = {
    'mcp': [0, 2],
    'pip_2': [2, 4],
    'pip_3': [4, 6],
    'pip_4': [6, 8],
    'pip_5': [8, 10],
    'mcp_1': [10, 12],
    'mcp_2': [12, 14],
    'mcp_3': [14, 16],
    'mcp_4': [16, 18],
    'mcp_5': [18, 20]
}

wrist_coord_mapping = {
    'wrist': [20, 26]
}

hand_outcome_mapping = {
    'mcp': [[], ['{part}_mcp_E__ip'] ],
    'pip_2': [['{part}_pip_J__2'], ['{part}_pip_E__2']],
    'pip_3': [['{part}_pip_J__3'], ['{part}_pip_E__3']],
    'pip_4': [['{part}_pip_J__4'], ['{part}_pip_E__4']],
    'pip_5': [['{part}_pip_J__5'], ['{part}_pip_E__5']],
    'mcp_1': [['{part}_mcp_J__1'], ['{part}_mcp_E__1']],
    'mcp_2': [['{part}_mcp_J__2'], ['{part}_mcp_E__2']],
    'mcp_3': [['{part}_mcp_J__3'], ['{part}_mcp_E__3']],
    'mcp_4': [['{part}_mcp_J__4'], ['{part}_mcp_E__4']],
    'mcp_5': [['{part}_mcp_J__5'], ['{part}_mcp_E__5']],
    "wrist": [['{part}_wrist_J__radcar', '{part}_wrist_J__mna', '{part}_wrist_J__cmc3', '{part}_wrist_J__capnlun', '{part}_wrist_J__cmc4', '{part}_wrist_J__cmc5'],
             ['{part}_wrist_E__mc1', '{part}_wrist_E__mul', '{part}_wrist_E__radius', '{part}_wrist_E__nav', '{part}_wrist_E__ulna', '{part}_wrist_E__lunate']]
}

dream_hand_parts = ['LH', 'RH']
dream_foot_parts = ['LF', 'RF']

class overall_joints_dataset(dream_dataset):
    def __init__(self, config, ds_type, cache_postfix = '', erosion_flag = False, pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, cache_postfix, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

        self.image_dir = config.train_fixed_location
        self.batch_size = 16

        self.ds_type = ds_type
        self.erosion_flag = erosion_flag
        if ds_type != 'test':
            if erosion_flag is False: 
                self.cache = self.cache + '_J'
                self.outcome = 'narrowing_sum'
            elif erosion_flag is True:
                self.cache = self.cache + '_E'
                self.outcome = 'erosion_sum'

            self.cache = self.cache + '_' + ds_type
        else:
            self.cache = True
            self.outcome = None

    def _create_overall_joints_dataset(self, outcomes_source, outcome_mapping, parts, joints_source, coord_keys):
        joints_df = pd.read_csv(joints_source)

        if outcomes_source is not None:
            intermediate_outcomes_df = self._create_intermediate_outcomes_df(outcomes_source, outcome_mapping, parts, joints_source)

            joints_df = joints_df.merge(intermediate_outcomes_df, on = ['image_name'])

        dataset = self._prepare_dataset(joints_df, coord_keys)

        return dataset

    def _create_intermediate_outcomes_df(self, outcomes_source, outcome_mapping, parts, joints_source):
        outcomes_df = pd.read_csv(outcomes_source)

        outcome_joints = []
        for _, row in outcomes_df.iterrows():
            patient_id = row['Patient_ID']

            for part in parts:
                image_name = patient_id + '-' + part

                narrowing_sum = 0
                erosion_sum = 0

                outcome_joint = {
                    'image_name': image_name,
                }

                for _, key in enumerate(outcome_mapping.keys()):
                    joint_mapping = outcome_mapping[key]

                    mapped_narrowing_keys = [key_val.format(part = part) for key_val in joint_mapping[0]]
                    for _, mapped_key in enumerate(mapped_narrowing_keys):
                        narrowing_sum += row[mapped_key]

                    mapped_erosion_keys = [key_val.format(part = part) for key_val in joint_mapping[1]]
                    for _, mapped_key in enumerate(mapped_erosion_keys):
                        erosion_sum += row[mapped_key]

                outcome_joint['narrowing_sum'] = narrowing_sum
                outcome_joint['erosion_sum'] = erosion_sum
                    
                outcome_joints.append(outcome_joint)

        return pd.DataFrame(outcome_joints, index = np.arange(len(outcome_joints)))

    def _prepare_dataset(self, outcome_joint_df, coord_keys):
        self.no_samples = outcome_joint_df.shape[0]

        outcome_joint_df = outcome_joint_df.sample(frac = 1).reset_index(drop = True)

        file_info = outcome_joint_df[['image_name', 'file_type', 'flip']].to_numpy()
        joint_coords = outcome_joint_df[coord_keys].to_numpy()
        outcomes = self._get_outcomes(outcome_joint_df)
        
        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords, outcomes))
        dataset = self._load_images(dataset)

        shuffle_and_augment = self.ds_type == 'train'

        dataset = self._cache_shuffle_repeat_dataset(dataset, cache = self.cache, buffer_size = 300, do_shuffle = shuffle_and_augment)
        if shuffle_and_augment:
            dataset = self._augment_images(dataset)

        return dataset

    def _finalize_dataset(self, dataset):
        if self.ds_type != 'test':
            dataset = self._remove_file_info(dataset)

            dataset = ds_ops.batch_and_prefetch_dataset(dataset, self.batch_size)
        else:
            dataset = self._remove_outcome(dataset)
            dataset = dataset.prefetch(buffer_size = AUTOTUNE)  

        return dataset

    def _get_outcomes(self, df, outcome_column = None):
        if self.outcome is not None:
            outcomes = df[self.outcome].to_numpy(dtype = np.float32)
        else:
            outcomes = np.zeros(df.shape[0], dtype = np.float32)

        return outcomes

    def _load_images(self, dataset):
        def __load_images(file_info, coords, outcomes):
            img, _ = img_ops.load_image(file_info, [], self.image_dir, imagenet = self.imagenet)

            return file_info, img, coords, outcomes

        return dataset.map(__load_images, num_parallel_calls = AUTOTUNE)

    def _augment_images(self, dataset):
        def __augment_images(file_info, img, coords, outcomes):
            img, coords = ds_ops._augment_and_clip_image(img, coords, update_labels = True)

            return file_info, img, coords, outcomes

        return dataset.map(__augment_images, num_parallel_calls = AUTOTUNE)

    def _load_joint(self, img, joint_key, coords):
        joint_img = js_ops._extract_joint_from_image(img, joint_key, coords[0], coords[1], self.joint_extractor)
        joint_img = img_ops.resize_image(joint_img, [], self.joint_height, self.joint_width, pad_resize = self.pad_resize)

        return joint_img

    def _remove_file_info(self, dataset):
        def _remove_file_info(file_info, img, y):
            return img, y

        return dataset.map(_remove_file_info, num_parallel_calls = AUTOTUNE)

    def _remove_outcome(self, dataset):
        def __remove_outcome(file_info, img, y):
            return file_info, img

        return dataset.map(__remove_outcome, num_parallel_calls = AUTOTUNE)

class hands_overall_joints_dataset(overall_joints_dataset):
    def __init__(self, config, ds_type, erosion_flag = False, pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, ds_type, cache_postfix = 'hands_overall_joints', erosion_flag = erosion_flag, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

    def create_hands_overall_joints_dataset(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_v2.csv'):
        dataset = self._create_overall_joints_dataset(outcomes_source, hand_outcome_mapping, dream_hand_parts, joints_source, hand_coord_keys)
        dataset = self._load_hand_joints(dataset)
        dataset = self._finalize_dataset(dataset)

        return dataset

    def create_hands_overall_joints_dataset_with_validation(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_train_v2.csv', joints_val_source = './data/predictions/hand_joint_data_test_v2.csv'):
        dataset = self.create_hands_overall_joints_dataset(outcomes_source, joints_source = joints_source)

        validation_dataset = self._create_validation_dataset()
        val_dataset = validation_dataset.create_hands_overall_joints_dataset(outcomes_source, joints_source = joints_val_source)

        return dataset, val_dataset, validation_dataset.no_samples

    def _load_hand_joints(self, dataset):
        def __load_hand_joints(file_info, img, coords, y):
            joints = []

            for joint_key in hand_coord_mapping.keys():
                # Skip MCP for narrowing
                if not self.erosion_flag and joint_key == 'mcp':
                    continue
                
                coord_idx = hand_coord_mapping[joint_key]

                joint_img = self._load_joint(img, joint_key, coords[coord_idx[0]:coord_idx[1]])

                joints.append(joint_img)

            wrist_coords = wrist_coord_mapping['wrist']
            wrist = self._load_wrist(img, coords[wrist_coords[0]:wrist_coords[1]])
            joints.append(wrist)

            return file_info, tuple(joints), y

        return dataset.map(__load_hand_joints, num_parallel_calls = AUTOTUNE)

    def _load_wrist(self, img, coords):
        wrist_img = js_ops._extract_wrist_from_image(img, coords[0], coords[2], coords[4], coords[1], coords[3], coords[5])
        wrist_img = img_ops.resize_image(wrist_img, [], self.joint_height, self.joint_width, pad_resize = self.pad_resize)

        return wrist_img

    def _create_validation_dataset(self):
        return hands_overall_joints_dataset(self.config, 'val', erosion_flag = self.erosion_flag, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor, imagenet = self.imagenet)