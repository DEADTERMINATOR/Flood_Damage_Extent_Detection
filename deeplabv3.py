import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision.transforms import Compose, Resize, v2
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

import os
import time
import random

import dataset
import dataloader


class DeepLabV3(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(DeepLabV3, self).__init__()
        self.deeplabv3_weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        self.resnet101_weights = models.ResNet101_Weights.DEFAULT
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(weights=self.deeplabv3_weights, weights_backbone=self.resnet101_weights)
        
        #Replaces the first convolution of the backbone of the model to accept 6-channel input.
        self.deeplabv3.backbone.conv1 = nn.Conv2d(num_input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        #Replaces the final classifier to change the number of output classes to 4.
        self.deeplabv3.classifier[-1] = torch.nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.deeplabv3.forward(x)
        return x
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
def visualize_results(num_results, predictions, images=None, masks=None, randomize_images=False):
    fig, axs = plt.subplots(num_results, 3, figsize=(32, 32))

    predictions_flat = [item for sublist in predictions for item in sublist]
    if (images != None):
        images_flat = [item for sublist in images for item in sublist]
    if (masks != None):
        masks_flat = [item for sublist in masks for item in sublist]
        
    print(len(predictions_flat))
    if (randomize_images):
        # Choose num_results number of images at random from the results.
        image_idxs = random.sample(range(0, len(predictions_flat) - 1), num_results)
    else:
        image_idxs = [i for i in range(num_results + 1)]
        
    for i in range(num_results):
        # Plot the input image and ground truth mask
        if (images == None or masks == None):    
            image, mask = test_dataset.get_item_no_transforms(image_idxs[i])
            
            axs[i, 0].imshow(image.numpy()[0:3, :, :].T, aspect='equal')
            axs[i, 0].imshow(image.numpy()[3:6, :, :].T, alpha=0.5, aspect='equal')
            axs[i, 2].imshow(mask.numpy().T, cmap="viridis", aspect='equal')
        else:
            image = images_flat[image_idxs[i]]
            mask = masks_flat[image_idxs[i]]
            
            axs[i, 0].imshow(image[0:3, :, :].T, aspect='equal')
            axs[i, 0].imshow(image[3:6, :, :].T, alpha=0.5, aspect='equal')
            axs[i, 2].imshow(mask.T, cmap="viridis", aspect='equal')

        axs[i, 0].set_title("Combined Image")
        axs[i, 0].axis('off')
        
        axs[i, 2].set_title("Ground Truth Mask")
        axs[i, 2].axis('off')
        
        # Plot the predicted image
        axs[i, 1].imshow(predictions_flat[image_idxs[i]].T, cmap="viridis", aspect='equal')
        axs[i, 1].set_title("Predicted Image")
        axs[i, 1].axis('off')

    plt.show()
    
batch_size = 8
num_input_channels = 6
num_classes = 4
lr = 1e-5
image_size = 224
# Whether the models parameters should be saved following the completion of a run.
save = True
#Whether an existing models parameters should be loaded before the run.
load = False

horizontal_flip = v2.RandomHorizontalFlip(p=0.5)
vertical_flip = v2.RandomVerticalFlip(p=0.5)
rotation = v2.RandomRotation(random.randint(1, 359))
random_crop = v2.RandomResizedCrop(size=image_size, antialias=True)

image_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(image_size, antialias=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  #These are the normalization values used by the pretrained weights in DeepLabv3
    #horizontal_flip,
    #vertical_flip,
    #rotation,
    #random_crop
    ])
mask_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.int64, scale=False),
    v2.Resize(image_size, antialias=True)
    #horizontal_flip,
    #vertical_flip,
    #rotation,
    #random_crop
    ])

cwd = os.getcwd()

train_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/training'), image_transforms=image_transforms, mask_transforms=mask_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/testing'), image_transforms=image_transforms, mask_transforms=mask_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DeepLabV3(num_input_channels, num_classes)
if (load):
    if (os.path.exists('DeepLabv3.pt')):
        print("Loading model.")
        model.load_state_dict(torch.load('DeepLabv3.pt'))
    else:
        print('Could not load model. File does not exist.')
model.to(device)
#model_preprocess = model.deeplabv3_weights.transforms()

criterion = torch.nn.CrossEntropyLoss()#FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

softmax = nn.Softmax(dim=1)

num_epochs = 50

images = []
masks = []
predicted_images = []

#Training
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_dataloader):
        image, mask = data
        
        image = image.to(device)
        mask = mask.squeeze().to(device)
        
        outputs = softmax(model(image)['out'])
        
        loss = criterion(outputs, mask)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()

        print('Batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Average Loss: %.4f' % (epoch + 1, num_epochs, epoch_loss / train_dataset.__len__()))
    
    total_loss = 0.0
    correct_predictions = 0
    total_pixels = 0
    dice_score = 0
 
#Testing
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            image, mask = data
            
            image = image.to(device)
            mask = mask.squeeze().to(device)
            
            outputs = softmax(model(image)['out'])
        
            loss = criterion(outputs, mask)
            total_loss += loss.item()
            
            predicted = torch.argmax(outputs, dim=1, keepdim=True)
            
            if (epoch + 1 == num_epochs):
                images.append(image.cpu().numpy())
                masks.append(mask.cpu().numpy())
                predicted_images.append(predicted.cpu().numpy())
            
            correct_predictions += (predicted == mask).sum().item()
            total_pixels += torch.numel(mask)
            dice_score += (2 * (predicted * mask).sum()) / ((predicted + mask).sum() + 1e-8)
    
    accuracy = correct_predictions / total_pixels * 100
    average_loss = total_loss / len(test_dataset)
    dice_score = dice_score / len(test_dataset)

    print('Accuracy: %.4f ---- Loss: %.4f ---- Dice: %.4f' % (accuracy, average_loss, dice_score))
    
    if (epoch + 1 == num_epochs):
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time at Epoch {epoch + 1} : {elapsed_time} seconds")
        
        if save:
            torch.save(model.state_dict(), 'DeepLabv3.pt')
            
        visualize_results(6, predicted_images, images, masks)
        
        images.clear()
        masks.clear()
        predicted_images.clear()
        
        start_time = time.time()