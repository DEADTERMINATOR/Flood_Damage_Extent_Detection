import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor

import numpy as np
import os

from PIL import Image


class HarveyData(Dataset):
    #dataset_dir: Provide a path to either "./dataset/training" or "./dataset/testing"
    #transforms: Any transformations that should be performed on the image when retrieved.
    def __init__(self, dataset_dir, image_transforms=None, mask_transforms=None):
        super(HarveyData, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        
        self.pre_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'pre_img')))
        self.post_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'post_img')))
        self.mask_paths = sorted(os.listdir(os.path.join(dataset_dir, 'post_msk')))
        
        self.pre_images = []
        self.post_images = []
        self.masks = []
        
        self.num_images = len(self.pre_image_paths)
        
        for i in range(self.num_images):
            pre_image = Image.open(os.path.join(dataset_dir, 'pre_img', self.pre_image_paths[i]))
            post_image = Image.open(os.path.join(dataset_dir, 'post_img', self.post_image_paths[i]))
            mask = Image.open(os.path.join(dataset_dir, 'post_msk', self.mask_paths[i]))
            
            self.pre_images.append(pre_image)
            self.post_images.append(post_image)
            self.masks.append(mask)
            
    def __getitem__(self, idx):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
            
        r, g, b, _ = mask.split()
        mask = Image.merge("RGBA", (r, g, b, r))
        mask = np.array(mask)
            
        #Apply transformations to images
        if (self.image_transforms is not None):
            pre_image = self.image_transforms(pre_image)
            post_image = self.image_transforms(post_image)
        if (self.mask_transforms is not None):
            mask = self.mask_transforms(mask)
            
        #Concatenate the pre and post disaster images together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image], dim=0)
        return combined_image, mask
        
    def get_item_no_transforms(self, idx):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
            
        #Convert image to normalized tensor.
        pre_image = to_tensor(pre_image)
        post_image = to_tensor(post_image)
        mask = to_tensor(mask)
        mask *= 255  # Manually adjust the label values back to the original values after the normalization of to_tensor()
        
        #Concatenate the pre and post disaster images together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image], dim=0)
        return combined_image, mask
    
    def __len__(self):
        return self.num_images
