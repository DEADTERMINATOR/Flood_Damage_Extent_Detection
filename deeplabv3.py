import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.transforms import Compose, Resize, v2
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import os
import time
import dataset
import dataloader


class DeepLabV3(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(DeepLabV3, self).__init__()
        self.deeplabv3_weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        self.resnet101_weights = models.ResNet101_Weights.DEFAULT
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(weights=self.deeplabv3_weights, weights_backbone=self.resnet101_weights)
        
        self.deeplabv3.backbone.conv1 = nn.Conv2d(num_input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.deeplabv3.classifier[-1] = torch.nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.deeplabv3.forward(x)
        return x
    
def visualize_results(num_results, predictions, images=None, masks=None):
    fig, axs = plt.subplots(num_results, 3, figsize=(32, 32))

    predictions_flat = [item for sublist in predictions for item in sublist]
    images_flat = [item for sublist in images for item in sublist]
    masks_flat = [item for sublist in masks for item in sublist]
        
    for i in range(num_results):
        # Plot the input image and ground truth mask
        if (images == None or masks == None):    
            image, mask = test_dataset.get_item_no_transforms(i)
            
            axs[i, 0].imshow(image.numpy()[3:6, :, :].T, aspect='equal')
            axs[i, 2].imshow(mask.numpy()[0].T, cmap="viridis", aspect='equal')
        else:
            image = images_flat[i]
            mask = masks_flat[i]
            
            axs[i, 0].imshow(image[3:6, :, :].T, aspect='equal')
            axs[i, 2].imshow(mask[0].T, cmap="viridis", aspect='equal')

        axs[i, 0].set_title("Post-Disaster Image")
        axs[i, 0].axis('off')
        
        axs[i, 2].set_title("Ground Truth Mask")
        axs[i, 2].axis('off')
        
        # Plot the predicted image
        axs[i, 1].imshow(predictions_flat[i].T, cmap="viridis", aspect='equal')
        axs[i, 1].set_title("Predicted Image")
        axs[i, 1].axis('off')

    plt.show()
    
batch_size = 8
num_input_channels = 6
num_classes = 4
lr = 1e-5
image_size = 224

image_transforms = v2.Compose([
    v2.Resize((image_size, image_size), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  #These are the normalization values used by the pretrained weights in DeepLabv3
    ])
mask_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    v2.Resize(image_size, antialias=True)
    ])

cwd = os.getcwd()

train_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/training'), image_transforms=image_transforms, mask_transforms=mask_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/testing'), image_transforms=image_transforms, mask_transforms=mask_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DeepLabV3(num_input_channels, num_classes).to(device)
#model_preprocess = model.deeplabv3_weights.transforms()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        mask = mask.to(device)
        
        optimizer.zero_grad()
        outputs = softmax(model(image)['out'])
        
        loss = criterion(outputs, mask)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()

        print('Batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Average Loss: %.4f' % (epoch + 1, num_epochs, epoch_loss / train_dataset.__len__()))
    
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_pixels = 0
    dice_score = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            image, mask = data
            
            image = image.to(device)
            mask = mask.to(device)
            
            outputs = softmax(model(image)['out'])
        
            loss = criterion(outputs, mask)
            total_loss += loss.item()
            
            predicted = torch.argmax(outputs, dim=1, keepdim=True)
            
            if (epoch + 1 == num_epochs):
                images.append(image.cpu().numpy())
                masks.append(mask.cpu().numpy())
                predicted_images.append(predicted.cpu().numpy())
            
            correct_predictions += (predicted == mask).sum()
            total_pixels += torch.numel(predicted)
            dice_score += (2 * (predicted * mask).sum()) / ((predicted + mask).sum() + 1e-8)
    
    accuracy = correct_predictions / total_pixels * 100
    average_loss = total_loss / len(test_dataset)
    dice_score = dice_score / len(test_dataset)

    print('Accuracy: %.4f ---- Loss: %.4f ---- Dice: %.4f' % (accuracy, average_loss, dice_score))
    if (epoch + 1 == num_epochs):
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time at Epoch {epoch + 1} : {elapsed_time} seconds")
        
        visualize_results(6, predicted_images, images, masks)
        
        images.clear()
        masks.clear()
        predicted_images.clear()
        
        start_time = time.time()