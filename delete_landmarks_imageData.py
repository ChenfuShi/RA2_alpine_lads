import json
import os
import pickle


PATH = 'C:\\Users\\CrankMuffler\\Documents\\GitHub\\RA2_alpine_lads'
os.chdir(PATH)

source_dir = 'C:\\Users\\CrankMuffler\\Development\\Dream\\training_v2020_01_13\\Feet\\landmarks'

dir_path = '.\\data\\landmarks'

landmark_files = os.listdir(source_dir)

for landmark_file in landmark_files:
    landmark_file_path = os.path.join(source_dir, landmark_file)
        
    with open(landmark_file_path, 'r') as landmark_json_file:
        landmark_json = json.load(landmark_json_file)
        
        landmark_json['imageData'] = ''
        
        with open(os.path.join(dir_path, landmark_file), 'w') as new_file:
            json.dump(landmark_json, new_file)