from re import A
import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision.transforms import Compose, Resize, v2
from torch.utils.data import DataLoader, Dataset

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

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
        
        #Replaces the final classifier to change the number of output classes to the required number of classes.
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
    fig, axes = plt.subplots(num_results, 3, figsize=(32, 32))

    predictions_flat = [item for sublist in predictions for item in sublist]
    if (images != None):
        images_flat = [item for sublist in images for item in sublist]
    if (masks != None):
        masks_flat = [item for sublist in masks for item in sublist]
        
    if (randomize_images):
        # Choose num_results number of images at random from the results.
        image_idxs = random.sample(range(0, len(predictions_flat) - 1), num_results)
    else:
        image_idxs = [i for i in range(1, num_results + 2)]
        
    for i in range(num_results):
        # Plot the input image and ground truth mask
        if (images == None or masks == None):    
            image, mask = test_dataset.get_item_resize_only(image_idxs[i], image_size)
            
            #Reorder the channels for matplotlib.
            image = torch.permute(image, (1, 2, 0))
            mask = torch.permute(mask, (1, 2, 0))
            
            axes[i, 0].imshow(image.numpy()[:, :, 0:3], aspect='equal')
            axes[i, 0].imshow(image.numpy()[:, :, 3:6], alpha=0.5, aspect='equal')
            axes[i, 0].imshow(image.numpy()[:, :, 6:7], alpha=0.5, aspect='equal')
            axes[i, 0].imshow(image.numpy()[:, :, 7:8], alpha=0.5, aspect='equal')
            axes[i, 2].imshow(mask.numpy(), cmap="viridis", aspect='equal')
        else:
            image = images_flat[image_idxs[i]]
            mask = masks_flat[image_idxs[i]]
            
            #Reorder the channels for matplotlib.
            image = np.transpose(image, (1, 2, 0))
            #mask = np.transpose(mask, (1, 2, 0))
            
            axes[i, 0].imshow(image[:, :, 0:3], aspect='equal')
            axes[i, 0].imshow(image[:, :, 3:6], alpha=0.5, aspect='equal')
            axes[i, 0].imshow(image[:, :, 6:7], alpha=0.5, aspect='equal')
            axes[i, 0].imshow(image[:, :, 7:8], alpha=0.5, aspect='equal')
            axes[i, 2].imshow(mask, cmap="viridis", aspect='equal')

        axes[i, 0].set_title("Combined Image")
        axes[i, 0].axis('off')
        
        axes[i, 2].set_title("Ground Truth Mask")
        axes[i, 2].axis('off')
        
        # Plot the predicted image
        axes[i, 1].imshow(predictions_flat[image_idxs[i]], cmap="viridis", aspect='equal')
        axes[i, 1].set_title("Predicted Image")
        axes[i, 1].axis('off')

    plt.show()

batch_size = 8
num_input_channels = 11
num_classes = 5
lr = 1e-4
image_size = 224

# Whether the models parameters should be saved following the completion of a run.
save = False
#Whether an existing models parameters should be loaded before the run.
load = False
#Tracking the highest current F1 score and the epoch it occurred on for the model to know when to best save the model.
highest_f1 = (0, 0)

cwd = os.getcwd()

print("Loading train and test images")

loading_start_time = time.time()
train_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/training'), image_size=image_size, verbose_logging=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
loading_end_time = time.time() - loading_start_time
print(f'Total Train Image Loading Time: {loading_end_time} seconds')

loading_start_time = time.time()
test_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/testing'), image_size=image_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
loading_end_time = time.time() - loading_start_time
print(f'Total Test Image Loading Time: {loading_end_time} seconds')

print("Finished loading images")

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

criterion = FocalLoss(reduction='sum')#torch.nn.CrossEntropyLoss()
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
    
    total_macro_precision = 0.0
    total_macro_recall = 0.0
    total_macro_f1 = 0.0
    
    total_class_precision = [0.0, 0.0, 0.0, 0.0, 0.0]
    total_class_recall = [0.0, 0.0, 0.0, 0.0, 0.0]
    total_class_f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
 
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
            
            predicted = torch.argmax(outputs, dim=1, keepdim=False)
            
            image = image.cpu().numpy()
            mask = mask.cpu().numpy()
            predicted = predicted.cpu().numpy()

            for i in range(len(mask)):
                # Calculate scores globally.
                precision, recall, f1, _ = precision_recall_fscore_support(mask[i].flatten(), predicted[i].flatten(), average='macro', zero_division=0.0)
                total_macro_precision += precision
                total_macro_recall += recall
                total_macro_f1 += f1
                
                # Calculate scores by class.
                precision, recall, f1, _ = precision_recall_fscore_support(mask[i].flatten(), predicted[i].flatten(), labels=[0, 1, 2, 3, 4], average=None, zero_division=0.0)
                total_class_precision += precision
                total_class_recall += recall
                total_class_f1 += f1
                      
            if (epoch + 1 == num_epochs):
                images.append(image)
                masks.append(mask)
                predicted_images.append(predicted)
    
    average_loss = total_loss / len(test_dataset)
    
    average_macro_precision = total_macro_precision / len(test_dataset)
    average_macro_recall = total_macro_recall / len(test_dataset)
    average_macro_f1 = total_macro_f1 / len(test_dataset)
    
    average_class_precision = total_class_precision / len(test_dataset)
    average_class_recall = total_class_recall / len(test_dataset)
    average_class_f1 = total_class_f1 / len(test_dataset)

    print('Average Macro Precision: %.4f ---- Average Macro Recall: %.4f ---- Average F1 Score: %.4f ---- Average Loss: %.4f' % (average_macro_precision, average_macro_recall, average_macro_f1, average_loss))
    print('Average No Damage Precision: %.4f ---- Average No Damage Recall: %.4f ---- Average No Damage F1: %.4f' % (average_class_precision[0], average_class_recall[0], average_class_f1[0]))
    print('Average Minor Precision: %.4f ---- Average Minor Recall: %.4f ---- Average Minor F1: %.4f' % (average_class_precision[1], average_class_recall[1], average_class_f1[1]))
    print('Average Moderate Precision: %.4f ---- Average Moderate Recall: %.4f ---- Average Moderate F1: %.4f' % (average_class_precision[2], average_class_recall[2], average_class_f1[2]))
    print('Average Major Precision: %.4f ---- Average Major Recall: %.4f ---- Average Major F1: %.4f' % (average_class_precision[3], average_class_recall[3], average_class_f1[3]))
    print('Average Background Precision: %.4f ---- Average Background Recall: %.4f ---- Average Background F1: %.4f' % (average_class_precision[4], average_class_recall[4], average_class_f1[4]))
    
    if (save and average_macro_f1 > highest_f1):
        highest_f1[0] = average_macro_f1
        highest_f1[1] = epoch + 1
        torch.save(model.state_dict(), 'DeepLabv3.pt')
        
    if (epoch + 1 == num_epochs):
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time at Epoch {epoch + 1} : {elapsed_time} seconds")
            
        if (save):
            print(f"Best Model with F1 Score of {highest_f1[0]} Saved on Epoch {highest_f1[1]}")
            
        visualize_results(6, predicted_images, images, masks)
        
        images.clear()
        masks.clear()
        predicted_images.clear()
        
        start_time = time.time()