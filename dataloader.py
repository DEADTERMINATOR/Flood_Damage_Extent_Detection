import torch
from torch.onnx import select_model_mode_for_export
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, hflip, vflip, rotate, adjust_gamma
from torchvision.transforms import Compose, Resize, v2

import numpy as np
import os
import random

from PIL import Image
import rasterio

import time


class HarveyData(Dataset):
    #dataset_dir: Provide a path to either "./dataset/training" or "./dataset/testing"
    #transforms: Any transformations that should be performed on the image when retrieved.
    def __init__(self, dataset_dir, image_size = 224, augment_data=True, verbose_logging=False):
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
        self.distance_coast_paths = sorted(os.listdir(os.path.join(dataset_dir, 'distance_to_coast')))
        self.distance_stream_paths = sorted(os.listdir(os.path.join(dataset_dir, 'distance_to_stream')))
        
        self.rain_824_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/824')))
        self.rain_825_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/825')))
        self.rain_826_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/826')))
        self.rain_827_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/827')))
        self.rain_828_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/828')))
        self.rain_829_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/829')))
        self.rain_830_paths = sorted(os.listdir(os.path.join(dataset_dir, 'rain/830')))
        
        self.stream_elev_824_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/824')))
        self.stream_elev_825_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/825')))
        self.stream_elev_826_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/826')))
        self.stream_elev_827_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/827')))
        self.stream_elev_828_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/828')))
        self.stream_elev_829_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/829')))
        self.stream_elev_830_paths = sorted(os.listdir(os.path.join(dataset_dir, 'stream_elev/830')))
        
        self.pre_images = []
        self.post_images = []
        self.masks = []
        
        self.elevation = []
        self.hand = []
        self.imperviousness = []
        self.distance_coast = []
        self.distance_stream = []

        self.rain_824 = []
        self.rain_825 = []
        self.rain_826 = []
        self.rain_827 = []
        self.rain_828 = []
        self.rain_829 = []
        self.rain_830 = []

        self.stream_elev_824 = []
        self.stream_elev_825 = []
        self.stream_elev_826 = []
        self.stream_elev_827 = []
        self.stream_elev_828 = []
        self.stream_elev_829 = []
        self.stream_elev_830 = []
        
        self.num_images = len(self.pre_image_paths)
        
        for i in range(self.num_images):
            pre_image = Image.open(os.path.join(dataset_dir, 'pre_img', self.pre_image_paths[i]))
            if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'pre_img', self.pre_image_paths[i]))
            
            post_image = Image.open(os.path.join(dataset_dir, 'post_img', self.post_image_paths[i]))
            if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'post_img', self.post_image_paths[i]))
            
            mask = Image.open(os.path.join(dataset_dir, 'PDE_labels', self.mask_paths[i])).convert('L')
            if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'PDE_labels', self.mask_paths[i]))
            
            with rasterio.open(os.path.join(dataset_dir, 'elevation', self.elevation_paths[i])) as src:
                elevation = src.read(1)
                elevation = torch.tensor(elevation).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'elevation', self.elevation_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'hand', self.hand_paths[i])) as src:
                hand = src.read(1)
                hand = torch.tensor(hand).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'hand', self.hand_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'imperviousness', self.imperviousness_paths[i])) as src:
                imperviousness = src.read(1)
                imperviousness = torch.tensor(imperviousness).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'imperviousness', self.imperviousness_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'distance_to_coast', self.distance_coast_paths[i])) as src:
                distance_coast = src.read(1)
                distance_coast = torch.tensor(distance_coast).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'distance_to_coast', self.distance_coast_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'distance_to_stream', self.distance_stream_paths[i])) as src:
                distance_stream = src.read(1)
                distance_stream = torch.tensor(distance_stream).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'distance_to_stream', self.distance_stream_paths[i]))

            with rasterio.open(os.path.join(dataset_dir, 'rain/824', self.rain_824_paths[i])) as src:
                rain_824 = src.read(1)
                rain_824 = torch.tensor(rain_824).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/824', self.rain_824_paths[i]))

            with rasterio.open(os.path.join(dataset_dir, 'rain/825', self.rain_825_paths[i])) as src:
                rain_825 = src.read(1)
                rain_825 = torch.tensor(rain_825).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/825', self.rain_825_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'rain/826', self.rain_826_paths[i])) as src:
                rain_826 = src.read(1)
                rain_826 = torch.tensor(rain_826).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/826', self.rain_826_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'rain/827', self.rain_827_paths[i])) as src:
                rain_827 = src.read(1)
                rain_827 = torch.tensor(rain_827).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/827', self.rain_827_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'rain/828', self.rain_828_paths[i])) as src:
                rain_828 = src.read(1)
                rain_828 = torch.tensor(rain_828).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/828', self.rain_828_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'rain/829', self.rain_829_paths[i])) as src:
                rain_829 = src.read(1)
                rain_829 = torch.tensor(rain_829).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/829', self.rain_829_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'rain/830', self.rain_830_paths[i])) as src:
                rain_830 = src.read(1)
                rain_830 = torch.tensor(rain_830).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'rain/830', self.rain_830_paths[i]))

            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_824_paths[i])) as src:
                stream_elev_824 = src.read(1)
                stream_elev_824 = torch.tensor(stream_elev_824).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_824_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/825', self.stream_elev_825_paths[i])) as src:
                stream_elev_825 = src.read(1)
                stream_elev_825 = torch.tensor(stream_elev_825).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_825_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/826', self.stream_elev_826_paths[i])) as src:
                stream_elev_826 = src.read(1)
                stream_elev_826 = torch.tensor(stream_elev_826).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_826_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/827', self.stream_elev_827_paths[i])) as src:
                stream_elev_827 = src.read(1)
                stream_elev_827 = torch.tensor(stream_elev_827).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_827_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/828', self.stream_elev_828_paths[i])) as src:
                stream_elev_828 = src.read(1)
                stream_elev_828 = torch.tensor(stream_elev_828).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_828_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/829', self.stream_elev_829_paths[i])) as src:
                stream_elev_829 = src.read(1)
                stream_elev_829 = torch.tensor(stream_elev_829).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_829_paths[i]))
                
            with rasterio.open(os.path.join(dataset_dir, 'stream_elev/830', self.stream_elev_830_paths[i])) as src:
                stream_elev_830 = src.read(1)
                stream_elev_830 = torch.tensor(stream_elev_830).unsqueeze(0)
                if verbose_logging: print("Loading " + os.path.join(dataset_dir, 'stream_elev/824', self.stream_elev_830_paths[i]))
            
            self.pre_images.append(pre_image)
            self.post_images.append(post_image)
            self.masks.append(mask)
            
            self.elevation.append(elevation)
            self.hand.append(hand)
            self.imperviousness.append(imperviousness)
            self.distance_coast.append(distance_coast)
            self.distance_stream.append(distance_stream)

            self.rain_824.append(rain_824)
            self.rain_825.append(rain_825)
            self.rain_826.append(rain_826)
            self.rain_827.append(rain_827)
            self.rain_828.append(rain_828)
            self.rain_829.append(rain_829)
            self.rain_830.append(rain_830)
            
            self.stream_elev_824.append(stream_elev_824)
            self.stream_elev_825.append(stream_elev_825)
            self.stream_elev_826.append(stream_elev_826)
            self.stream_elev_827.append(stream_elev_827)
            self.stream_elev_828.append(stream_elev_828)
            self.stream_elev_829.append(stream_elev_829)
            self.stream_elev_830.append(stream_elev_830)
            
    def __getitem__(self, idx):
        #Get pre and post image, and the mask, for the current index.
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        mask = self.masks[idx]
        
        elevation = self.elevation[idx]
        hand = self.hand[idx]
        imperviousness = self.imperviousness[idx]
        distance_coast = self.distance_coast[idx]
        distance_stream = self.distance_stream[idx]
        
        rain_824 = self.rain_824[idx]
        rain_825 = self.rain_825[idx]
        rain_826 = self.rain_826[idx]
        rain_827 = self.rain_827[idx]
        rain_828 = self.rain_828[idx]
        rain_829 = self.rain_829[idx]
        rain_830 = self.rain_830[idx]
        
        stream_elev_824 = self.stream_elev_824[idx]
        stream_elev_825 = self.stream_elev_825[idx]
        stream_elev_826 = self.stream_elev_826[idx]
        stream_elev_827 = self.stream_elev_827[idx]
        stream_elev_828 = self.stream_elev_828[idx]
        stream_elev_829 = self.stream_elev_829[idx]
        stream_elev_830 = self.stream_elev_830[idx]
            
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
        distance_coast = meta_transforms(distance_coast)
        distance_stream = meta_transforms(distance_stream)

        rain_824 = meta_transforms(rain_824)
        rain_825 = meta_transforms(rain_825)
        rain_826 = meta_transforms(rain_826)
        rain_827 = meta_transforms(rain_827)
        rain_828 = meta_transforms(rain_828)
        rain_829 = meta_transforms(rain_829)
        rain_830 = meta_transforms(rain_830)

        stream_elev_824 = meta_transforms(stream_elev_824)
        stream_elev_825 = meta_transforms(stream_elev_825)
        stream_elev_826 = meta_transforms(stream_elev_826)
        stream_elev_827 = meta_transforms(stream_elev_827)
        stream_elev_828 = meta_transforms(stream_elev_828)
        stream_elev_829 = meta_transforms(stream_elev_829)
        stream_elev_830 = meta_transforms(stream_elev_830)
            
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
                distance_coast = vflip(distance_coast)
                distance_stream = vflip(distance_stream)
                
                rain_824 = vflip(rain_824)
                rain_825 = vflip(rain_825)
                rain_826 = vflip(rain_826)
                rain_827 = vflip(rain_827)
                rain_828 = vflip(rain_828)
                rain_829 = vflip(rain_829)
                rain_830 = vflip(rain_830)
                
                stream_elev_824 = vflip(stream_elev_824)
                stream_elev_825 = vflip(stream_elev_825)
                stream_elev_826 = vflip(stream_elev_826)
                stream_elev_827 = vflip(stream_elev_827)
                stream_elev_828 = vflip(stream_elev_828)
                stream_elev_829 = vflip(stream_elev_829)
                stream_elev_830 = vflip(stream_elev_830)
                
                mask = vflip(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 1:
                # flip image horizontally
                pre_image = hflip(pre_image)
                post_image = hflip(post_image)
                
                elevation = hflip(elevation)
                hand = hflip(hand)
                imperviousness = hflip(imperviousness)
                distance_coast = hflip(distance_coast)
                distance_stream = hflip(distance_stream)
                
                rain_824 = hflip(rain_824)
                rain_825 = hflip(rain_825)
                rain_826 = hflip(rain_826)
                rain_827 = hflip(rain_827)
                rain_828 = hflip(rain_828)
                rain_829 = hflip(rain_829)
                rain_830 = hflip(rain_830)
                
                stream_elev_824 = hflip(stream_elev_824)
                stream_elev_825 = hflip(stream_elev_825)
                stream_elev_826 = hflip(stream_elev_826)
                stream_elev_827 = hflip(stream_elev_827)
                stream_elev_828 = hflip(stream_elev_828)
                stream_elev_829 = hflip(stream_elev_829)
                stream_elev_830 = hflip(stream_elev_830)
                
                mask = hflip(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 2:
                # zoom image
                zoom = v2.RandomResizedCrop(self.size, antialias=True)

                pre_image = zoom(pre_image)
                post_image = zoom(post_image)
                
                elevation = zoom(elevation)
                hand = zoom(hand)
                imperviousness = zoom(imperviousness)
                distance_coast = zoom(distance_coast)
                distance_stream = zoom(distance_stream)
                
                rain_824 = zoom(rain_824)
                rain_825 = zoom(rain_825)
                rain_826 = zoom(rain_826)
                rain_827 = zoom(rain_827)
                rain_828 = zoom(rain_828)
                rain_829 = zoom(rain_829)
                rain_830 = zoom(rain_830)
                
                stream_elev_824 = zoom(stream_elev_824)
                stream_elev_825 = zoom(stream_elev_825)
                stream_elev_826 = zoom(stream_elev_826)
                stream_elev_827 = zoom(stream_elev_827)
                stream_elev_828 = zoom(stream_elev_828)
                stream_elev_829 = zoom(stream_elev_829)
                stream_elev_830 = zoom(stream_elev_830)
                
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
                distance_coast = adjust_gamma(distance_coast)
                distance_stream = adjust_gamma(distance_stream)
                
                rain_824 = adjust_gamma(rain_824)
                rain_825 = adjust_gamma(rain_825)
                rain_826 = adjust_gamma(rain_826)
                rain_827 = adjust_gamma(rain_827)
                rain_828 = adjust_gamma(rain_828)
                rain_829 = adjust_gamma(rain_829)
                rain_830 = adjust_gamma(rain_830)
                
                stream_elev_824 = adjust_gamma(stream_elev_824)
                stream_elev_825 = adjust_gamma(stream_elev_825)
                stream_elev_826 = adjust_gamma(stream_elev_826)
                stream_elev_827 = adjust_gamma(stream_elev_827)
                stream_elev_828 = adjust_gamma(stream_elev_828)
                stream_elev_829 = adjust_gamma(stream_elev_829)
                stream_elev_830 = adjust_gamma(stream_elev_830)
                
                mask = adjust_gamma(mask, gamma)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 4:
                # perform elastic transformation
                elastic = v2.ElasticTransform(sigma=10)

                pre_image = elastic(pre_image)
                post_image = elastic(post_image)
                
                elevation = elastic(elevation)
                hand = elastic(hand)
                imperviousness = elastic(imperviousness)
                distance_coast = elastic(distance_coast)
                distance_stream = elastic(distance_stream)
                
                rain_824 = elastic(rain_824)
                rain_825 = elastic(rain_825)
                rain_826 = elastic(rain_826)
                rain_827 = elastic(rain_827)
                rain_828 = elastic(rain_828)
                rain_829 = elastic(rain_829)
                rain_830 = elastic(rain_830)
                
                stream_elev_824 = elastic(stream_elev_824)
                stream_elev_825 = elastic(stream_elev_825)
                stream_elev_826 = elastic(stream_elev_826)
                stream_elev_827 = elastic(stream_elev_827)
                stream_elev_828 = elastic(stream_elev_828)
                stream_elev_829 = elastic(stream_elev_829)
                stream_elev_830 = elastic(stream_elev_830)
                
                mask = elastic(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 5:
                # modify brightness/contrast/saturation/hue
                jitter = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)

                pre_image = jitter(pre_image)
                post_image = jitter(post_image)
                
                elevation = jitter(elevation)
                hand = jitter(hand)
                imperviousness = jitter(imperviousness)
                distance_coast = jitter(distance_coast)
                distance_stream = jitter(distance_stream)
                
                rain_824 = jitter(rain_824)
                rain_825 = jitter(rain_825)
                rain_826 = jitter(rain_826)
                rain_827 = jitter(rain_827)
                rain_828 = jitter(rain_828)
                rain_829 = jitter(rain_829)
                rain_830 = jitter(rain_830)
                
                stream_elev_824 = jitter(stream_elev_824)
                stream_elev_825 = jitter(stream_elev_825)
                stream_elev_826 = jitter(stream_elev_826)
                stream_elev_827 = jitter(stream_elev_827)
                stream_elev_828 = jitter(stream_elev_828)
                stream_elev_829 = jitter(stream_elev_829)
                stream_elev_830 = jitter(stream_elev_830)
                
                mask = jitter(mask)
            elif augment_mode_1 or augment_mode_2 or augment_mode_3 == 6:
                # rotate image
                random_degree = random.randint(1, 359)

                pre_image = rotate(pre_image, random_degree)
                post_image = rotate(post_image, random_degree)
                
                elevation = rotate(elevation, random_degree)
                hand = rotate(hand, random_degree)
                imperviousness = rotate(imperviousness, random_degree)
                distance_coast = rotate(distance_coast)
                distance_stream = rotate(distance_stream)
                
                rain_824 = rotate(rain_824)
                rain_825 = rotate(rain_825)
                rain_826 = rotate(rain_826)
                rain_827 = rotate(rain_827)
                rain_828 = rotate(rain_828)
                rain_829 = rotate(rain_829)
                rain_830 = rotate(rain_830)
                
                stream_elev_824 = rotate(stream_elev_824)
                stream_elev_825 = rotate(stream_elev_825)
                stream_elev_826 = rotate(stream_elev_826)
                stream_elev_827 = rotate(stream_elev_827)
                stream_elev_828 = rotate(stream_elev_828)
                stream_elev_829 = rotate(stream_elev_829)
                stream_elev_830 = rotate(stream_elev_830)
                
                mask = rotate(mask, random_degree)
                
        #Concatenate the pre and post disaster images, as well as the meta-attributes, together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image, elevation, """hand""", imperviousness, distance_coast, distance_stream], dim=0)
                                    #rain_824, rain_825, rain_826, rain_827, rain_828, rain_829, rain_830], dim=0)
                                    #stream_elev_824, stream_elev_825, stream_elev_826, stream_elev_827, stream_elev_828, stream_elev_829, stream_elev_830], dim=0)
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
        distance_coast = to_tensor(distance_coast)
        distance_stream = to_tensor(distance_stream)
                
        rain_824 = to_tensor(rain_824)
        rain_825 = to_tensor(rain_825)
        rain_826 = to_tensor(rain_826)
        rain_827 = to_tensor(rain_827)
        rain_828 = to_tensor(rain_828)
        rain_829 = to_tensor(rain_829)
        rain_830 = to_tensor(rain_830)
                
        stream_elev_824 = to_tensor(stream_elev_824)
        stream_elev_825 = to_tensor(stream_elev_825)
        stream_elev_826 = to_tensor(stream_elev_826)
        stream_elev_827 = to_tensor(stream_elev_827)
        stream_elev_828 = to_tensor(stream_elev_828)
        stream_elev_829 = to_tensor(stream_elev_829)
        stream_elev_830 = to_tensor(stream_elev_830)
        
        #Resize the images to the same size as was used during training.
        resize = v2.Compose([v2.Resize((image_size, image_size), antialias=True)])
        pre_image = resize(pre_image)
        post_image = resize(post_image)
        mask = resize(mask)
        
        elevation = resize(elevation)
        hand = resize(hand)
        imperviousness = resize(imperviousness)
        distance_coast = resize(distance_coast)
        distance_stream = resize(distance_stream)
                
        rain_824 = resize(rain_824)
        rain_825 = resize(rain_825)
        rain_826 = resize(rain_826)
        rain_827 = resize(rain_827)
        rain_828 = resize(rain_828)
        rain_829 = resize(rain_829)
        rain_830 = resize(rain_830)
                
        stream_elev_824 = resize(stream_elev_824)
        stream_elev_825 = resize(stream_elev_825)
        stream_elev_826 = resize(stream_elev_826)
        stream_elev_827 = resize(stream_elev_827)
        stream_elev_828 = resize(stream_elev_828)
        stream_elev_829 = resize(stream_elev_829)
        stream_elev_830 = resize(stream_elev_830)
        
        #Concatenate the pre and post disaster images, as well as the meta attributes, together along the channel dimension.
        combined_image = torch.cat([pre_image, post_image, elevation, imperviousness, distance_coast, distance_stream], dim=0)
        return combined_image, mask
    
    def __len__(self):
        return self.num_images
