from dataset.joint_dataset import feet_joint_dataset, hands_joints_dataset, hands_wrists_dataset, joint_narrowing_dataset
from dataset.test_dataset import joint_test_dataset, narrowing_test_dataset

class hands_joints_val_dataset(hands_joints_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

    def create_hands_joints_dataset_with_validation(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_train_v2.csv', joints_val_source = './data/predictions/hand_joint_data_test_v2.csv', erosion_flag = False):
        dataset = self.create_hands_joints_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)
        
        val_dataset, val_no_samples = self._create_test_dataset().get_hands_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)

class hands_wrists_val_dataset(hands_wrists_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)
        
    def create_wrists_joints_dataset_with_validation(self, outcomes_source, joints_source = './data/predictions/hand_joint_data_train_v2.csv', joints_val_source = './data/predictions/hand_joint_data_test_v2.csv', erosion_flag = False):
        dataset = self.create_wrists_joints_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)
        
        val_dataset, val_no_samples = self._create_test_dataset().get_wrists_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor, imagenet = self.imagenet)
    
class feet_joint_val_dataset(feet_joint_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None, imagenet = False):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor, imagenet = imagenet)

    def create_feet_joints_dataset_with_validation(self, outcomes_source, joints_source = './data/predictions/feet_joint_data_train_v2.csv', joints_val_source = './data/predictions/feet_joint_data_test_v2.csv', erosion_flag = False):
        dataset = self.create_feet_joints_dataset(outcomes_source, joints_source = joints_source, erosion_flag = erosion_flag)
        
        val_dataset, val_no_samples = self._create_test_dataset().get_feet_joint_test_dataset(joints_source = joints_val_source, outcomes_source = outcomes_source, erosion_flag = erosion_flag)

        return dataset, val_dataset, val_no_samples
    
    def _create_test_dataset(self):
        return joint_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)
    
class joint_narrowing_val_dataset(joint_narrowing_dataset):
    def __init__(self, config, model_type = 'R', pad_resize = False, joint_extractor = None):
        super().__init__(config, model_type = model_type, pad_resize = pad_resize, joint_extractor = joint_extractor)
        
    def create_combined_narrowing_joint_dataset_with_validation(self, outcomes_source, 
            hand_joints_source = './data/predictions/hand_joint_data_train_v2.csv', hand_joints_val_source = './data/predictions/hand_joint_data_test_v2.csv', 
            feet_joints_source = './data/predictions/feet_joint_data_train_v2.csv', feet_joints_val_source = './data/predictions/feet_joint_data_test_v2.csv'):

        dataset = self.create_combined_narrowing_joint_dataset(outcomes_source, hand_joints_source = hand_joints_source, feet_joints_source = feet_joints_source)
        
        test_dataset = narrowing_test_dataset(self.config, self.image_dir, model_type = self.model_type, pad_resize = self.pad_resize, joint_extractor = self.joint_extractor)
        val_dataset, val_no_samples = test_dataset.get_joint_narrowing_test_dataset(hand_joints_source = hand_joints_val_source,
            feet_joints_source = feet_joints_val_source, outcomes_source = outcomes_source)

        return dataset, val_dataset, val_no_samples