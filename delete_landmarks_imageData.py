import json
import os
import pickle


PATH = 'C:\\Users\\CrankMuffler\\Documents\\GitHub\\RA2_alpine_lads'
os.chdir(PATH)

source_dir = 'C:\\Users\\CrankMuffler\\Documents\\GitHub\\RA2_alpine_lads\\data\\landmarks'

dir_path = '.\\data\\landmarks\\hands'

landmark_files = os.listdir(source_dir)

for landmark_file in landmark_files:
    landmark_file_path = os.path.join(source_dir, landmark_file)
    
    if 'F.json' not in landmark_file:
        if landmark_file.endswith('.json'):
            with open(landmark_file_path, 'r') as landmark_json_file:
                landmark_json = json.load(landmark_json_file)

                landmark_json['imageData'] = ''

                with open(os.path.join(dir_path, landmark_file), 'w') as new_file:
                    json.dump(landmark_json, new_file)