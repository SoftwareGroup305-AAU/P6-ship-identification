import os
import random
import string

# Paths to your image and label folders
IMAGE_DIR = 'yolo/data2/train/images'
LABEL_DIR = 'yolo/data2/train/labels'

# Length of the random string
RANDOM_NAME_LENGTH = 12

def generate_random_string(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def rename_images_and_labels(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    for image_file in image_files:
        name, ext = os.path.splitext(image_file)
        label_file = name + '.txt'
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # Generate new name
        new_base = generate_random_string(RANDOM_NAME_LENGTH)
        new_image_path = os.path.join(image_dir, new_base + ext)
        new_label_path = os.path.join(label_dir, new_base + '.txt')

        try:
            os.rename(image_path, new_image_path)
            if os.path.exists(label_path):
                os.rename(label_path, new_label_path)
                print(f"Renamed: {image_file} + {label_file} ➜ {new_base}")
            else:
                print(f"Image '{image_file}' has no matching label. Renamed image only.")
        except Exception as e:
            print(f"Error renaming {image_file}: {e}")

if __name__ == '__main__':
    rename_images_and_labels(IMAGE_DIR, LABEL_DIR)
