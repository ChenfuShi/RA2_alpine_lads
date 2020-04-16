from dataset.joints import joint_extractor

def get_joint_extractor(joint_type, erosion_flag):
    extractor = joint_extractor.default_joint_extractor()
    
    if joint_type == 'H' and not erosion_flag:
        extractor = joint_extractor.width_based_joint_extractor()
    elif joint_type == 'F' and not erosion_flag:
        extractor = joint_extractor.width_based_joint_extractor(joint_scale = 5., height_scale = 0.9, key_joint_scales = {'mtp': 4., 'mtp_1': 4.})
    elif joint_type == "H" and erosion_flag:
        extractor = joint_extractor.width_based_joint_extractor(joint_scale = 4.5, height_scale = 1.2)
    elif joint_type == "F" and erosion_flag:
        extractor = joint_extractor.width_based_joint_extractor(joint_scale = 4.2, height_scale = 1.2, key_joint_scales = {"mtp":3.2,"mtp_1":3.2}, key_height_scales = {})
    elif joint_type == "RSNA" and erosion_flag:
        extractor = joint_extractor.width_based_joint_extractor(joint_scale = 5., height_scale = 1.2, key_joint_scales = {}, key_height_scales = {})
    elif joint_type == "RSNA" and not erosion_flag:
        extractor = joint_extractor.width_based_joint_extractor(joint_scale = 6.5, height_scale = 0.8, key_joint_scales = {}, key_height_scales = {})
    return extractor
