supported_file_types = ['jpg', 'png', 'jpeg']

def is_supported_image_file(file_parts):
    if(len(file_parts) > 1 and file_parts[1] in supported_file_types):
        return True
    else:
        return False