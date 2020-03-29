from dataset.joints import joint_extractor

def get_joint_extractor(joint_type, erosion_flag):
    extractor = joint_extractor.default_joint_extractor()
    
    if joint_type == 'H' and not erosion_flag:
        extractor =  joint_extractor.width_based_joint_extractor()
        
    return extractor