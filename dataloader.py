import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, hflip, vflip, rotate, adjust_gamma
from torchvision.transforms import Compose, Resize, v2

import numpy as np
import os
import random

from PIL import Image


class HarveyData(Dataset):
    #dataset_dir: Provide a path to either "./dataset/training" or "./dataset/testing"
    #transforms: Any transformations that should be performed on the image when retrieved.
    def __init__(self, dataset_dir, image_transforms=None, mask_transforms=None, augment_data=True):
        super(HarveyData, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.augment_data = augment_data
        
        self.pre_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'pre_img')))
        self.post_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'post_img')))
        self.mask_paths = sorted(os.listdir(os.path.join(dataset_dir, 'PDE_labels')))
        
        self.pre_images = []
        self.post_images = []
        self.masks = []
        
        self.num_images = len(self.pre_image_paths)
        
        for i in range(self.num_images):
            pre_image = Image.open(os.path.join(dataset_dir, 'pre_img', self.pre_image_paths[i]))
            post_image = Image.open(os.path.join(dataset_dir, 'post_img', self.post_image_paths[i]))
            mask = Image.open(os.path.join(dataset_dir, 'PDE_labels', self.mask_paths[i])).convert('L')
            
            self.pre_images.append(pre_image)
            self.post_images.append(post_image)
            self.masks.append(mask)
            
    def __getitem__(self, idx):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
            
        #Apply transformations to images
        if (self.image_transforms is not None):
            pre_image = self.image_transforms(pre_image)
            post_image = self.image_transforms(post_image)
        if (self.mask_transforms is not None):
            mask = self.mask_transforms(mask)
            
        if self.augment_data:
            augmentation_switches = {0, 1, 2, 3, 4, 5, 6}
            augment_mode_1 = np.random.choice(list(augmentation_switches))
            augmentation_switches.remove(augment_mode_1)

            additional_augment_chance = np.random.random()
            augment_mode_2 = -1
            augment_mode_3 = -1

            if (additional_augment_chance > 0.5):
                augment_mode_2 = np.random.choice(list(augmentation_switches))
                augmentation_switches.remove(augment_mode_2)
            #if (additional_augment_chance > 0.8):
                #augment_mode_3 = np.random.choice(list(augmentation_switches))
                #augmentation_switches.remove(augment_mode_3)

            if augment_mode_1 or augment_mode_2 or augment_mode_3 == 0:
                # flip image vertically
                pre_image = vflip(pre_image)
                post_image = vflip(post_image)
                mask = vflip(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 1:
                # flip image horizontally
                pre_image = hflip(pre_image)
                post_image = hflip(post_image)
                mask = hflip(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 2:
                # zoom image
                zoom = v2.RandomResizedCrop(self.size, antialias=True)

                pre_image = zoom(pre_image)
                post_image = zoom(post_image)
                mask = zoom(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 3:
                # modify gamma
                min_gamma = 0.25
                gamma_range = 2.25
                gamma = gamma_range * np.random.random() + min_gamma

                pre_image = adjust_gamma(pre_image, gamma)
                post_image = adjust_gamma(post_image, gamma)
                mask = adjust_gamma(mask, gamma)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 4:
                # perform elastic transformation
                elastic = v2.ElasticTransform(sigma=10)

                pre_image = elastic(pre_image)
                post_image = elastic(post_image)
                mask = elastic(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 5:
                # modify brightness/contrast/saturation/hue
                jitter = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)

                pre_image = jitter(pre_image)
                post_image = jitter(post_image)
                mask = jitter(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 6:
                # rotate image
                random_degree = random.randint(1, 359)

                pre_image = rotate(pre_image, random_degree)
                post_image = rotate(post_image, random_degree)
                mask = rotate(mask, random_degree)
                
        #Concatenate the pre and post disaster images together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image], dim=0)
        return combined_image, mask
        
    def get_item_resize_only(self, idx, image_size):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
            
        #Convert image to normalized tensor.
        pre_image = to_tensor(pre_image)
        post_image = to_tensor(post_image)
        mask = to_tensor(mask)
        mask *= 255  # Manually adjust the label values back to the original values after the normalization of to_tensor()
        
        #Resize the images to the same size as was used during training.
        resize = v2.Compose([v2.Resize((image_size, image_size), antialias=True)])
        pre_image = resize(pre_image)
        post_image = resize(post_image)
        mask = resize(mask)
        
        #Concatenate the pre and post disaster images together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image], dim=0)
        return combined_image, mask
    
    def __len__(self):
        return self.num_images
