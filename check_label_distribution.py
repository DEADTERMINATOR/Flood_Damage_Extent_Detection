import os
import numpy as np
from PIL import Image

complete_label_path = 'dataset/PDE_labels'
train_label_path = 'DAHitRa/data/harvey/train/PDE_labels'
val_label_path = 'DAHitRa/data/harvey/val/PDE_labels'
test_label_path = 'DAHitRa/data/harvey/test/PDE_labels'

#label_paths = [complete_label_path]
label_paths = [train_label_path, val_label_path, test_label_path]

def get_labels_from_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)
    labels = [tuple(color) for color in unique_colors]
    
    return labels

for label_path in label_paths:
    label_images = os.listdir(label_path)

    num_no_damage_samples = 0
    num_minor_damage_samples = 0
    num_moderate_damage_samples = 0
    num_major_damage_samples = 0
    num_background_samples = 0
    
    for label_image in label_images:
        image_labels = get_labels_from_image(os.path.join(label_path, label_image))
        
        if (0,0,0) in image_labels:
            num_no_damage_samples += 1
        if (1,1,1) in image_labels:
            num_minor_damage_samples +=1
        if (2,2,2) in image_labels:
            num_moderate_damage_samples +=1
            print("Moderate Damage Image:", label_image)
        if (3,3,3) in image_labels:
            num_major_damage_samples += 1
            print("Major Damage Image:", label_image)
        if (4,4,4) in image_labels:
            num_background_samples += 1
            
    print("No Damage:", num_no_damage_samples)
    print("Minor Damage:", num_minor_damage_samples)
    print("Moderate Damage:", num_moderate_damage_samples)
    print("Major Damage:", num_major_damage_samples)
    print("Background:", num_background_samples)
    print("")