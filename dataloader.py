import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor

import os

from PIL import Image


class HarveyData(Dataset):
    #dataset_dir: Provide a path to either "./dataset/training" or "./dataset/testing"
    #transforms: Any transformations that should be performed on the image when retrieved.
    def __init__(self, dataset_dir, transforms=None):
        super(HarveyData, self).__init__()
        self.transforms = transforms
        
        pre_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'pre_img')))
        post_image_paths = sorted(os.listdir(os.path.join(dataset_dir, 'post_img')))
        mask_paths = sorted(os.listdir(os.path.join(dataset_dir, 'post_msk')))
        
        self.pre_images = []
        self.post_images = []
        self.masks = []
        
        self.num_images = len(pre_image_paths)
        
        for i in range(self.num_images):
            pre_image = Image.open(os.path.join(dataset_dir, 'pre_img', pre_image_paths[i]))
            post_image = Image.open(os.path.join(dataset_dir, 'post_img', post_image_paths[i]))
            mask = Image.open(os.path.join(dataset_dir, 'post_msk', mask_paths[i]))
            
            self.pre_images.append(pre_image)
            self.post_images.append(post_image)
            self.masks.append(mask)
            
    def __getitem__(self, idx):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
            
        #Convert image to normalized tensor.
        pre_image = to_tensor(pre_image)
        post_image = to_tensor(post_image)
        
        #Convert mask to tensor without normalization.
        mask_to_tensor = transforms.Compose([transforms.PILToTensor()])
        mask = mask_to_tensor(mask)
            
        #Concatenate the pre and post disaster images together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image], dim=0)
            
        #Apply transformations to image
        if self.transforms is not None:
            self.transforms(combined_image)
            self.transforms(mask)

        return combined_image, mask
        
    def __len__(self):
        return self.num_images
