import os
import shutil
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from PIL import Image

# Paths setup
pre_image_path = 'dataset/pre_img'
label_path = 'dataset/PDE_labels'

post_img_path = 'dataset\post_img'

elevation_path = 'dataset\elevation'
hand_path = 'dataset\hand'
imperviousness_path = 'dataset\imperviousness'
distance_to_coast_path = 'dataset\distance_to_coast'
distance_to_stream_path = 'dataset\distance_to_stream'

rain_824_path = r'dataset\rain_824'
rain_825_path = r'dataset\rain_825'
rain_826_path = r'dataset\rain_826'
rain_827_path = r'dataset\rain_827'
rain_828_path = r'dataset\rain_828'
rain_829_path = r'dataset\rain_829'
rain_830_path = r'dataset\rain_830'

stream_elev_824_path = 'dataset\stream_elev_824'
stream_elev_825_path = 'dataset\stream_elev_825'
stream_elev_826_path = 'dataset\stream_elev_826'
stream_elev_827_path = 'dataset\stream_elev_827'
stream_elev_828_path = 'dataset\stream_elev_828'
stream_elev_829_path = 'dataset\stream_elev_829'
stream_elev_830_path = 'dataset\stream_elev_830'

attribute_dirs = [post_img_path, label_path, elevation_path, hand_path, imperviousness_path, distance_to_coast_path, distance_to_stream_path,
                  rain_824_path, rain_825_path, rain_826_path, rain_827_path, rain_828_path, rain_829_path, rain_830_path,
                  stream_elev_824_path, stream_elev_825_path, stream_elev_826_path, stream_elev_827_path, stream_elev_828_path, stream_elev_829_path, stream_elev_830_path]

output_train_path = 'dataset/train'
output_val_path = 'dataset/val'
output_test_path = 'dataset/test'

def get_labels_from_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)
    labels = [tuple(color) for color in unique_colors]
    
    return labels

# Modify move_images function to also handle attribute images
def move_images(image_filenames, destination):
    for filename in image_filenames.flatten():
        shutil.move(os.path.join(pre_image_path, filename), os.path.join(destination, 'pre_img', filename))
        print(f"Moving {filename} to {os.path.join(destination, 'pre_img', filename)}")
        for attr_dir in attribute_dirs:
            attr_name = os.path.basename(attr_dir)
            attr_filename = filename.replace('pre_img', attr_name)
            attr_src_path = os.path.join(attr_dir, attr_filename)
            attr_dst_path = os.path.join(destination, attr_name, attr_filename)
            shutil.move(attr_src_path, attr_dst_path)
            print(f"Moving {attr_filename} to {os.path.join(destination, attr_name, attr_filename)}")

images = os.listdir(pre_image_path)

labels = []
label_images = os.listdir(label_path)

print("Getting unique labels from each PDE label image.")
for label_image in label_images:
    image_labels = get_labels_from_image(os.path.join(label_path, label_image))
    labels.append(image_labels)
    print(f"Got unique labels from {label_image}")
print("Finished getting unique labels")

unique_labels = sorted(set(label for sublist in labels for label in sublist))

# Convert labels to a binary indicator matrix
y = np.array([[1 if unique_label in image_labels else 0 for unique_label in unique_labels] for image_labels in labels])

# Convert your images list to a numpy array and reshape for compatibility with iterative_train_test_split
X = np.array(images).reshape(-1, 1)

X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)
X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

move_images(X_train, 'dataset/train')
move_images(X_val, 'dataset/val')
move_images(X_test, 'dataset/test')

print('Images and attribute images have been moved to their respective directories.')