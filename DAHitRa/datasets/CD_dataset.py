"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
import cv2
import rasterio

from torch.utils import data

from datasets.data_utils import CDDataAugmentation, CDDataAugmentation_Harvey
from sklearn.model_selection import train_test_split

"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "images"
IMG_POST_FOLDER_NAME = 'images'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "targets"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir, split, img_name):
    # img_name = img_name + "_post_disaster.png"
    return os.path.join(root_dir, split, 'post_img', img_name)


def get_img_path(root_dir, split, img_name):
    # img_name = img_name + "_pre_disaster.png"
    return os.path.join(root_dir, split, 'pre_img', img_name)


def get_label_path(root_dir, split, img_name):
    #img_name = img_name + "_post_disaster.png"
    # img_name = img_name.replace("images", "masks") + "_post_disaster.png"
    return os.path.join(root_dir, split, 'PDE_labels', img_name)

class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        self.is_train = is_train
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = os.listdir(os.path.join(self.root_dir, split, 'pre_img'))
        self.post_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'post_img'))

        # self.img_name_list = load_img_name_list(self.list_path)
        # tmp_lst = [("hurricane" in x) for x in self.img_name_list]
        # self.img_name_list = np.array(self.img_name_list)[tmp_lst].tolist()
        
        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                with_random_resize=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.post_img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, patch=None):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        self.split = split
        self.patch = patch

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        
        L_path = get_label_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        label = np.array(Image.open(L_path), dtype=np.uint8)
        

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
           label = label // 255

        # # 2 class model
        # label[label <= 1] = 0
        # label[label > 1] = 1
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor, patch=self.patch)
        # [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor, split=self.split)
        return {'name': name, 'A': img, 'B': img_B, 'L': label}


class xBDataset(data.Dataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(xBDataset, self).__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.label_transform = label_transform
        self.split = split

        train_dirs = ['../data/train'] # fix path!!
        all_files = []
        for d in train_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
#                if ('_pre_disaster.png' in f) and (('hurricane-harvey' in f) | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f) | ('palu-tsunami' in     f)):
                 if ('_pre_disaster.png' in f):
                    all_files.append(os.path.join(d, 'images', f))

        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=10)
        if split == 'train':
            self.img_name_list = np.array(all_files)[train_idxs]
        elif split == 'val':
            self.img_name_list = np.array(all_files)[val_idxs]
        elif split == 'test':
            self.img_name_list = np.array(all_files)[val_idxs]


    def __getitem__(self, index):
        fn = self.img_name_list[index]
        
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)
        label = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        label[label <= 2] = 0
        label[label > 2] = 1
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        name = fn.split('/')[-1]
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)


class xBDatasetMulti(data.Dataset):
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(xBDatasetMulti, self).__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.label_transform = label_transform
        self.split = split

        train_dirs = ['../data/xbd/train'] # fix path!!
        all_files = []
        for d in train_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                if ('_pre_disaster.png' in f) and (('hurricane-harvey' in f) | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f) | ('palu-tsunami' in     f)):
                    all_files.append(os.path.join(d, 'images', f))

        # Upsampling
        file_classes = []
        for fn in all_files:
            fl = np.zeros((4,), dtype=bool)
            msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
            for c in range(1, 5):
                fl[c-1] = c in msk1
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)
        for i in range(len(file_classes)):
            im = all_files[i]
            if file_classes[i, 1:].max():
                all_files.append(im)
            if file_classes[i, 1:3].max():
                all_files.append(im)

        # train test split
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=10)
        if split == 'train':
            self.img_name_list = np.array(all_files)[train_idxs]
        elif split == 'val':
            self.img_name_list = np.array(all_files)[val_idxs]
        elif split == 'test':
            self.img_name_list = np.array(all_files)[val_idxs]


    def __getitem__(self, index):
        fn = self.img_name_list[index]
        
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)
        label = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)

        if self.split == 'train':
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor, is_train=True)
        else:
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor, is_train=False)

        name = fn.split('/')[-1]
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)
    
class HarveyDataset(ImageDataset):
    
    def load_meta_attributes_by_index(self, index):
        with rasterio.open(os.path.join(self.root_dir, self.split, 'elevation', self.elevation_img_name_list[index % self.A_size])) as src:
            elevation = src.read(1)
            elevation = np.expand_dims(elevation, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'imperviousness', self.imperviousness_img_name_list[index % self.A_size])) as src:
            imperviousness = src.read(1)
            imperviousness = np.expand_dims(imperviousness, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'hand', self.hand_img_name_list[index % self.A_size])) as src:
            hand = src.read(1)
            hand = np.expand_dims(hand, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'distance_to_coast', self.dist_coast_img_name_list[index % self.A_size])) as src:
            dist_coast = src.read(1)
            dist_coast = np.expand_dims(dist_coast, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'distance_to_stream', self.dist_stream_img_name_list[index % self.A_size])) as src:
            dist_stream = src.read(1)
            dist_stream = np.expand_dims(dist_stream, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/824', self.rain_824_img_name_list[index % self.A_size])) as src:
            rain_824 = src.read(1).astype(np.float32)
            rain_824 = np.expand_dims(rain_824, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/825', self.rain_825_img_name_list[index % self.A_size])) as src:
            rain_825 = src.read(1).astype(np.float32)
            rain_825 = np.expand_dims(rain_825, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/826', self.rain_826_img_name_list[index % self.A_size])) as src:
            rain_826 = src.read(1).astype(np.float32)
            rain_826 = np.expand_dims(rain_826, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/827', self.rain_827_img_name_list[index % self.A_size])) as src:
            rain_827 = src.read(1).astype(np.float32)
            rain_827 = np.expand_dims(rain_827, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/828', self.rain_828_img_name_list[index % self.A_size])) as src:
            rain_828 = src.read(1).astype(np.float32)
            rain_828 = np.expand_dims(rain_828, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/829', self.rain_829_img_name_list[index % self.A_size])) as src:
            rain_829 = src.read(1).astype(np.float32)
            rain_829 = np.expand_dims(rain_829, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'rain/830', self.rain_830_img_name_list[index % self.A_size])) as src:
            rain_830 = src.read(1).astype(np.float32)
            rain_830 = np.expand_dims(rain_830, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/824', self.stream_elev_824_img_name_list[index % self.A_size])) as src:
            stream_elev_824 = src.read(1).astype(np.float32)
            stream_elev_824 = np.expand_dims(stream_elev_824, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/825', self.stream_elev_825_img_name_list[index % self.A_size])) as src:
            stream_elev_825 = src.read(1).astype(np.float32)
            stream_elev_825 = np.expand_dims(stream_elev_825, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/826', self.stream_elev_826_img_name_list[index % self.A_size])) as src:
            stream_elev_826 = src.read(1).astype(np.float32)
            stream_elev_826 = np.expand_dims(stream_elev_826, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/827', self.stream_elev_827_img_name_list[index % self.A_size])) as src:
            stream_elev_827 = src.read(1).astype(np.float32)
            stream_elev_827 = np.expand_dims(stream_elev_827, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/828', self.stream_elev_828_img_name_list[index % self.A_size])) as src:
            stream_elev_828 = src.read(1).astype(np.float32)
            stream_elev_828 = np.expand_dims(stream_elev_828, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/829', self.stream_elev_829_img_name_list[index % self.A_size])) as src:
            stream_elev_829 = src.read(1).astype(np.float32)
            stream_elev_829 = np.expand_dims(stream_elev_829, axis=0)
            
        with rasterio.open(os.path.join(self.root_dir, self.split, 'stream_elev/830', self.stream_elev_830_img_name_list[index % self.A_size])) as src:
            stream_elev_830 = src.read(1).astype(np.float32)
            stream_elev_830 = np.expand_dims(stream_elev_830, axis=0)
           
        return {
            "elevation": elevation,
            "imperviousness": imperviousness,
            "hand": hand,
            "dist_coast": dist_coast,
            "dist_stream": dist_stream,
            "rain_824": rain_824,
            "rain_825": rain_825,
            "rain_826": rain_826,
            "rain_827": rain_827,
            "rain_828": rain_828,
            "rain_829": rain_829,
            "rain_830": rain_830,
            "stream_elev_824": stream_elev_824,
            "stream_elev_825": stream_elev_825,
            "stream_elev_826": stream_elev_826,
            "stream_elev_827": stream_elev_827,
            "stream_elev_828": stream_elev_828,
            "stream_elev_829": stream_elev_829,
            "stream_elev_830": stream_elev_830,
            }

        """
        return {
            self.elevation_img_name_list[index % self.A_size]: elevation,
            self.imperviousness_img_name_list[index % self.A_size]: imperviousness,
            self.hand_img_name_list[index % self.A_size]: hand,
            self.dist_coast_img_name_list[index % self.A_size]: dist_coast,
            self.dist_stream_img_name_list[index % self.A_size]: dist_stream,
            self.rain_824_img_name_list[index % self.A_size]: rain_824,
            self.rain_825_img_name_list[index % self.A_size]: rain_825,
            self.rain_826_img_name_list[index % self.A_size]: rain_826,
            self.rain_827_img_name_list[index % self.A_size]: rain_827,
            self.rain_828_img_name_list[index % self.A_size]: rain_828,
            self.rain_829_img_name_list[index % self.A_size]: rain_829,
            self.rain_830_img_name_list[index % self.A_size]: rain_830,
            self.stream_elev_824_img_name_list[index % self.A_size]: stream_elev_824,
            self.stream_elev_825_img_name_list[index % self.A_size]: stream_elev_825,
            self.stream_elev_826_img_name_list[index % self.A_size]: stream_elev_826,
            self.stream_elev_827_img_name_list[index % self.A_size]: stream_elev_827,
            self.stream_elev_828_img_name_list[index % self.A_size]: stream_elev_828,
            self.stream_elev_829_img_name_list[index % self.A_size]: stream_elev_829,
            self.stream_elev_830_img_name_list[index % self.A_size]: stream_elev_830,
            }
        """
            

    def __init__(self, root_dir, img_size, split='train', is_train=True,
                 to_tensor=True, load_meta_attributes_upfront=False, diff_block=0):
        super(HarveyDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.split = split
        self.load_meta_attributes_upfront = load_meta_attributes_upfront
        self.diff_block = diff_block
        
        self.label_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'PDE_labels'))
        self.elevation_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'elevation'))
        self.imperviousness_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'imperviousness'))
        self.hand_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'hand'))
        self.dist_coast_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'distance_to_coast'))
        self.dist_stream_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'distance_to_stream'))
        
        self.rain_824_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/824'))
        self.rain_825_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/825'))
        self.rain_826_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/826'))
        self.rain_827_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/827'))
        self.rain_828_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/828'))
        self.rain_829_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/829'))
        self.rain_830_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'rain/830'))
        
        self.stream_elev_824_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/824'))
        self.stream_elev_825_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/825'))
        self.stream_elev_826_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/826'))
        self.stream_elev_827_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/827'))
        self.stream_elev_828_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/828'))
        self.stream_elev_829_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/829'))
        self.stream_elev_830_img_name_list = os.listdir(os.path.join(self.root_dir, split, 'stream_elev/830'))
        
        """
        if (load_meta_attributes_upfront):
            self.elevation_imgs = []
            self.imperviousness_imgs = []
            self.hand_imgs = []
            self.dist_coast_imgs = []
            self.dist_stream_imgs = []
            
            self.rain_824_imgs = []
            self.rain_825_imgs = []
            self.rain_826_imgs = []
            self.rain_827_imgs = []
            self.rain_828_imgs = []
            self.rain_829_imgs = []
            self.rain_830_imgs = []

            self.stream_elev_824_imgs = []
            self.stream_elev_825_imgs = []
            self.stream_elev_826_imgs = []
            self.stream_elev_827_imgs = []
            self.stream_elev_828_imgs = []
            self.stream_elev_829_imgs = []
            self.stream_elev_830_imgs = []
            
            for i in range(len(self.rain_824_img_name_list)):
                attribute_dict = self.load_meta_attributes_by_index(i)
                self.elevation_imgs.append(attribute_dict["elevation"])
                self.imperviousness_imgs.append(attribute_dict["imperviousness"])
                self.hand_imgs.append(attribute_dict["hand"])
                self.dist_coast_imgs.append(attribute_dict["dist_coast"])
                self.dist_stream_imgs.append(attribute_dict["dist_stream"])
                
                self.rain_824_imgs.append(attribute_dict["rain_824"])
                self.rain_825_imgs.append(attribute_dict["rain_825"])
                self.rain_826_imgs.append(attribute_dict["rain_826"])
                self.rain_827_imgs.append(attribute_dict["rain_827"])
                self.rain_828_imgs.append(attribute_dict["rain_828"])
                self.rain_829_imgs.append(attribute_dict["rain_829"])
                self.rain_830_imgs.append(attribute_dict["rain_830"])
                
                self.stream_elev_824_imgs.append(attribute_dict["stream_elev_824"])
                self.stream_elev_825_imgs.append(attribute_dict["stream_elev_825"])
                self.stream_elev_826_imgs.append(attribute_dict["stream_elev_826"])
                self.stream_elev_827_imgs.append(attribute_dict["stream_elev_827"])
                self.stream_elev_828_imgs.append(attribute_dict["stream_elev_828"])
                self.stream_elev_829_imgs.append(attribute_dict["stream_elev_829"])
                self.stream_elev_830_imgs.append(attribute_dict["stream_elev_830"])
           """
            
        if self.is_train:
            self.augm = CDDataAugmentation_Harvey(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_random_rot=True,
                with_random_crop=True
            )
        else:
            self.augm = CDDataAugmentation_Harvey(
                img_size=self.img_size
            )


    def __getitem__(self, index):            
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.post_img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        
        L_path = get_label_path(self.root_dir, self.split, self.label_img_name_list[index % self.A_size])
        label = np.array(Image.open(L_path).convert('L'), dtype=np.uint8)
        
        """
        if (self.load_meta_attributes_upfront):
            attribute_dict = {
                "elevation": self.elevation_imgs[index],
                "imperviousness": self.imperviousness_imgs[index],
                "hand": self.hand_imgs[index],
                "dist_coast": self.dist_coast_imgs[index],
                "dist_stream": self.dist_stream_imgs[index],
                "rain_824": self.rain_824_imgs[index],
                "rain_825": self.rain_825_imgs[index],
                "rain_826": self.rain_826_imgs[index],
                "rain_827": self.rain_827_imgs[index],
                "rain_828": self.rain_828_imgs[index],
                "rain_829": self.rain_829_imgs[index],
                "rain_830": self.rain_830_imgs[index],
                "stream_elev_824": self.stream_elev_824_imgs[index],
                "stream_elev_825": self.stream_elev_825_imgs[index],
                "stream_elev_826": self.stream_elev_826_imgs[index],
                "stream_elev_827": self.stream_elev_827_imgs[index],
                "stream_elev_828": self.stream_elev_828_imgs[index],
                "stream_elev_829": self.stream_elev_829_imgs[index],
                "stream_elev_830": self.stream_elev_830_imgs[index],
            }
        else:
        """
        attribute_dict = self.load_meta_attributes_by_index(index)
            
        if self.diff_block == 1 or self.diff_block == 3:
            [img, img_B], attributes, [label] = self.augm.transform(imgs=[img, img_B], labels=[label], attributes=attribute_dict, diff_block=self.diff_block)
            return {'name': name, 'A': img, 'B': img_B, 'C': attributes, 'L': label}
        else:
            [img, img_B], [label] = self.augm.transform(imgs=[img, img_B], labels=[label], attributes=attribute_dict, diff_block=self.diff_block)
            return {'name': name, 'A': img, 'B': img_B, 'L': label}
        
