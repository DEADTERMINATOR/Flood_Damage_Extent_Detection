import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, hflip, vflip, rotate, adjust_gamma
from torchvision.transforms import Compose, Resize, v2

import numpy as np
import os
import random

from PIL import Image
import rasterio


class HarveyData(Dataset):
    #dataset_dir: Provide a path to either "./dataset/training" or "./dataset/testing"
    #transforms: Any transformations that should be performed on the image when retrieved.
    def __init__(self, dataset_dir, image_size = 224, augment_data=True):
        super(HarveyData, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.augment_data = augment_data
        
        self.pre_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'pre_img')))
        self.post_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'post_img')))
        self.mask_paths = sorted(os.listdir(os.path.join(dataset_dir, 'PDE_labels')))
        self.elevation_paths = sorted(os.listdir(os.path.join(dataset_dir, 'elevation')))
        self.hand_paths = sorted(os.listdir(os.path.join(dataset_dir, 'hand')))
        self.imperviousness_paths = sorted(os.listdir(os.path.join(dataset_dir, 'imperviousness')))
        
        self.pre_images = []
        self.post_images = []
        self.masks = []
        
        self.elevation = []
        self.hand = []
        self.imperviousness = []
        
        self.num_images = len(self.pre_image_paths)
        
        for i in range(self.num_images):
            pre_image = Image.open(os.path.join(dataset_dir, 'pre_img', self.pre_image_paths[i]))
            post_image = Image.open(os.path.join(dataset_dir, 'post_img', self.post_image_paths[i]))
            mask = Image.open(os.path.join(dataset_dir, 'PDE_labels', self.mask_paths[i])).convert('L')
            
            with rasterio.open(os.path.join(dataset_dir, 'elevation', self.elevation_paths[i])) as src:
                elevation = src.read(1)
                elevation = torch.tensor(elevation).unsqueeze(0)
            with rasterio.open(os.path.join(dataset_dir, 'hand', self.hand_paths[i])) as src:
                hand = src.read(1)
                hand = torch.tensor(hand).unsqueeze(0)
            with rasterio.open(os.path.join(dataset_dir, 'imperviousness', self.imperviousness_paths[i])) as src:
                imperviousness = src.read(1)
                imperviousness = torch.tensor(imperviousness).unsqueeze(0)
            
            self.pre_images.append(pre_image)
            self.post_images.append(post_image)
            self.masks.append(mask)
            
            self.elevation.append(elevation)
            self.hand.append(hand)
            self.imperviousness.append(imperviousness)
            
    def __getitem__(self, idx):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
        
        elevation = self.elevation[idx]
        hand = self.hand[idx]
        imperviousness = self.imperviousness[idx]
            
        image_transforms = v2.Compose([
                           v2.ToImage(),
                           v2.ToDtype(torch.float32, scale=True),
                           v2.Resize((self.image_size, self.image_size), antialias=True),
                           v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  #These are the normalization values used by the pretrained weights in DeepLabv3
        ])
        mask_transforms = v2.Compose([
                          v2.ToImage(),
                          v2.ToDtype(torch.int64, scale=False),
                          v2.Resize((self.image_size, self.image_size), antialias=True)
        ])
        meta_transforms = v2.Compose([
                          v2.ToImage(),
                          v2.ToDtype(torch.float32, scale=True),
                          v2.Resize((self.image_size, self.image_size), antialias=True),
                          v2.Grayscale()
        ])
        
        pre_image = image_transforms(pre_image)
        post_image = image_transforms(post_image)
        mask = mask_transforms(mask)
        
        elevation = meta_transforms(elevation)
        hand = meta_transforms(hand)
        imperviousness = meta_transforms(imperviousness)
            
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
                
                elevation = vflip(elevation)
                hand = vflip(hand)
                imperviousness = vflip(imperviousness)
                
                mask = vflip(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 1:
                # flip image horizontally
                pre_image = hflip(pre_image)
                post_image = hflip(post_image)
                
                elevation = hflip(elevation)
                hand = hflip(hand)
                imperviousness = hflip(imperviousness)
                
                mask = hflip(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 2:
                # zoom image
                zoom = v2.RandomResizedCrop(self.size, antialias=True)

                pre_image = zoom(pre_image)
                post_image = zoom(post_image)
                
                elevation = zoom(elevation)
                hand = zoom(hand)
                imperviousness = zoom(imperviousness)
                
                mask = zoom(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 3:
                # modify gamma
                min_gamma = 0.25
                gamma_range = 2.25
                gamma = gamma_range * np.random.random() + min_gamma

                pre_image = adjust_gamma(pre_image, gamma)
                post_image = adjust_gamma(post_image, gamma)
                
                elevation = adjust_gamma(elevation)
                hand = adjust_gamma(hand)
                imperviousness = adjust_gamma(imperviousness)
                
                mask = adjust_gamma(mask, gamma)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 4:
                # perform elastic transformation
                elastic = v2.ElasticTransform(sigma=10)

                pre_image = elastic(pre_image)
                post_image = elastic(post_image)
                
                elevation = elastic(elevation)
                hand = elastic(hand)
                imperviousness = elastic(imperviousness)
                
                mask = elastic(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 5:
                # modify brightness/contrast/saturation/hue
                jitter = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)

                pre_image = jitter(pre_image)
                post_image = jitter(post_image)
                
                elevation = jitter(elevation)
                hand = jitter(hand)
                imperviousness = jitter(imperviousness)
                
                mask = jitter(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 6:
                # rotate image
                random_degree = random.randint(1, 359)

                pre_image = rotate(pre_image, random_degree)
                post_image = rotate(post_image, random_degree)
                
                elevation = rotate(elevation, random_degree)
                hand = rotate(hand, random_degree)
                imperviousness = rotate(imperviousness, random_degree)
                
                mask = rotate(mask, random_degree)
                
        #Concatenate the pre and post disaster images, as well as the meta-attributes, together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image, elevation, imperviousness], dim=0)
        return combined_image, mask
        
    def get_item_resize_only(self, idx, image_size):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
        
        elevation = self.elevation[idx]
        hand = self.hand[idx]
        imperviousness = self.imperviousness[idx]
            
        #Convert image to normalized tensor.
        pre_image = to_tensor(pre_image)
        post_image = to_tensor(post_image)
        
        mask = to_tensor(mask)
        mask *= 255  # Manually adjust the label values back to the original values after the normalization of to_tensor()
        
        elevation = to_tensor(elevation)
        hand = to_tensor(hand)
        imperviousness = to_tensor(imperviousness)
        
        #Resize the images to the same size as was used during training.
        resize = v2.Compose([v2.Resize((image_size, image_size), antialias=True)])
        pre_image = resize(pre_image)
        post_image = resize(post_image)
        mask = resize(mask)
        
        elevation = resize(elevation)
        hand = resize(hand)
        imperviousness = resize(imperviousness)
        
        #Concatenate the pre and post disaster images, as well as the meta attributes, together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image, elevation, imperviousness], dim=0)
        return combined_image, mask
    
    def __len__(self):
        return self.num_images
