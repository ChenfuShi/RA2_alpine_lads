import pandas as pd
import tensorflow as tf

import dataset.ops.image_ops as img_ops
import dataset.joint_dataset as joint_dataset
import dataset.ops.joint_ops as joint_ops

AUTOTUNE = tf.data.experimental.AUTOTUNE

class joint_test_dataset(joint_dataset.joint_dataset):
    def __init__(self, config, img_dir):
        super().__init__(config)

        self.img_dir = img_dir

    def get_feet_joint_test_dataset(self, joints_source = './data/predictions/feet_joint_data_test.csv'):
        return self._create_joint_dataset(joints_source, joint_dataset.foot_outcome_mapping.keys())

    def get_hand_joint_test_dataset(self, joints_source = './data/predictions/hand_joint_data_test.csv'):
        return self._create_joint_dataset(joints_source, joint_dataset.hand_outcome_mapping.keys())

    def get_wrist_joint_testdataset(self, joints_source = './data/predictions/hand_joint_data_test.csv'):
        return self._create_joint_dataset(joints_source, joint_dataset.hand_joint_keys, load_wrists = True)

    def _create_joint_dataset(self, joints_source, joint_keys, load_wrists = False):
        if load_wrists:
            joints_df = self._create_intermediate_wrists_df(joints_source, joint_dataset.hand_wrist_keys)
        else:
            joints_df = self._create_intermediate_joints_df(joints_source, joint_keys)
        
        file_info = joints_df[['image_name', 'file_type', 'flip', 'key']].to_numpy()
        
        if load_wrists:
            joint_coords = joints_df[['w1_x', 'w1_y', 'w2_x', 'w2_y', 'w3_x', 'w3_y']].to_numpy()
        else:
            joint_coords = joints_df[['coord_x', 'coord_y']].to_numpy()

        dataset = tf.data.Dataset.from_tensor_slices((file_info, joint_coords))

        if load_wrists:
            dataset = self._load_wrists_without_outcomes(dataset)
        else:
            dataset = self._load_joints_without_outcomes(dataset)

        dataset = self._resize_images_without_outcomes(dataset)

        return dataset.prefetch(buffer_size = AUTOTUNE)
        
    def _load_joints_without_outcomes(self, dataset):
        def __load_joints(file_info, y):
            x_coord = y[0]
            y_coord = y[1]

            full_img, _ = img_ops.load_image(file_info, [], self.img_dir)

            joint_img = joint_ops._extract_joint_from_image(full_img, x_coord, y_coord)

            return file_info, joint_img

        return dataset.map(__load_joints, num_parallel_calls=AUTOTUNE)

    def _load_wrists_without_outcomes(self, dataset):
        def __load_wrists(file_info, y):
            w1_x = y[0]
            w2_x = y[2]
            w3_x = y[4]
            w1_y = y[1]
            w2_y = y[3]
            w3_y = y[5]

            full_img, _ = img_ops.load_image(file_info, [], self.img_dir)

            joint_img = joint_ops._extract_wrist_from_image(full_img, w1_x, w2_x, w3_x, w1_y, w2_y, w3_y)

            return file_info, joint_img

        return dataset.map(__load_wrists, num_parallel_calls=AUTOTUNE)

    def _resize_images_without_outcomes(self, dataset):
        def __resize(file_info, img):
            img, _ =  img_ops.resize_image(img, [], self.config.joint_img_height, self.config.joint_img_width, pad_resize = False, update_labels = False)

            return file_info, img

        return dataset.map(__resize, num_parallel_calls=AUTOTUNE)