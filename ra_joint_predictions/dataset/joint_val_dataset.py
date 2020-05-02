from dataset.joint_dataset import feet_joint_dataset, hands_joints_dataset, hands_wrists_dataset, combined_joint_dataset, mixed_joint_dataset
from dataset.test_dataset import joint_test_dataset, combined_test_dataset

hands_joints_source = './data/predictions/hands_joint_data_train_010holdout.csv'
hands_joints_val_source = './data/predictions/hands_joint_data_test_010holdout.csv'

feet_joints_source = './data/predictions/feet_joint_data_train_010holdout.csv'
feet_joints_val_source = './data/predictions/feet_joint_data_test_010holdout.csv'

class hands_joints_val_dataset(hands_joints_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None, apply_clahe = False):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type, apply_clahe = apply_clahe)

    def create_hands_joints_dataset_with_validation(self, outcomes_source, joints_source = hands_joints_source, joints_val_source = hands_joints_val_source, erosion_flag = False):
        dataset = self.create_hands_joints_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)
        
        val_dataset, val_no_samples = self._create_test_dataset().get_hands_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor, apply_clahe = self.apply_clahe)

class hands_wrists_val_dataset(hands_wrists_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)
        
    def create_wrists_joints_dataset_with_validation(self, outcomes_source, joints_source = hands_joints_source, joints_val_source = hands_joints_val_source, erosion_flag = False):
        dataset = self.create_wrists_joints_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)
        
        val_dataset, val_no_samples = self._create_test_dataset().get_wrists_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor, imagenet = self.imagenet)
    
class feet_joint_val_dataset(feet_joint_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False, split_type = None):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet, split_type = split_type)

    def create_feet_joints_dataset_with_validation(self, outcomes_source, joints_source = feet_joints_source, joints_val_source = feet_joints_val_source, erosion_flag = False):
        dataset = self.create_feet_joints_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)
        
        val_dataset, val_no_samples = self._create_test_dataset().get_feet_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)
    
class combined_joint_val_dataset(combined_joint_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor)
        
    def create_combined_joint_dataset_with_validation(self, outcomes_source, 
            hand_joints_source = hands_joints_source, hand_joints_val_source = hands_joints_val_source, 
            feet_joints_source = feet_joints_source, feet_joints_val_source = feet_joints_val_source, erosion_flag = False):

        dataset = self.create_combined_joint_dataset(outcomes_source, hand_joints_source = hand_joints_source, feet_joints_source = feet_joints_source, erosion_flag = erosion_flag)
        
        test_dataset = combined_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)
        val_dataset, val_no_samples = test_dataset.get_combined_joint_test_dataset(hand_joints_source = hand_joints_val_source,
            feet_joints_source = feet_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
class mixed_joint_val_dataset(mixed_joint_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, joint_type = 'MH', split_type = None):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, joint_type = joint_type, split_type = split_type)
        
    def create_mixed_joint_val_dataset_with_validation(self, outcomes_source, 
            hand_joints_source = hands_joints_source, hand_joints_val_source = hands_joints_val_source, 
            feet_joints_source = feet_joints_source, feet_joints_val_source = feet_joints_val_source, erosion_flag = False):
        
        dataset = self.create_mixed_joint_dataset(outcomes_source, hand_joints_source = hand_joints_source, feet_joints_source = feet_joints_source, erosion_flag = erosion_flag)
        
        if self.is_main_hand:
            val_dataset, val_no_samples = self._create_test_dataset().get_hands_joint_test_dataset(joints_source = hand_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)
        else:
            val_dataset, val_no_samples = self._create_test_dataset().get_feet_joint_test_dataset(joints_source = feet_joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)
            
        return dataset, val_dataset, val_no_samples
            
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor) 
