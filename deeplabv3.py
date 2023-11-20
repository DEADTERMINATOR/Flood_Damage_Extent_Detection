import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.transforms import Compose, Resize, v2
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import os
import time
import dataset
import dataloader


class DeepLabV3(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(DeepLabV3, self).__init__()
        self.deeplabv3_weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        self.resnet50_weights = models.ResNet50_Weights.DEFAULT
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes, weights_backbone=self.resnet50_weights)
        
        # Modify the first convolutional layer of the ResNet50 backbone to accept num_input_channels channels        
        backbone_conv1 = nn.Conv2d(num_input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modified_backbone = nn.Sequential(backbone_conv1,
                                              *list(self.deeplabv3.backbone.children())[1:])
        self.deeplabv3_head = nn.Sequential(*list(self.deeplabv3.children())[-1:])
        self.deeplabv3_combine = nn.Sequential(self.modified_backbone,
                                               self.deeplabv3_head)

        #print("----------ORIGINAL----------")
        #print(nn.Sequential(*list(self.deeplabv3.children())[-2:]))
        #print("----------MODIFIED----------")
        #print(self.deeplabv3_combine)
        #print("----------LAST ITEM----------")
        #print(nn.Sequential(*list(self.deeplabv3.children())[-1:]))
        
    def forward(self, x):
        #x = self.deeplabv3.forward(x)
        x = self.deeplabv3_combine.forward(x)
        return x

batch_size = 8
num_input_channels = 6
num_classes = 4
lr = 1e-5
image_size = 224

transforms = v2.Compose([
    v2.Resize((image_size, image_size), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(1, 359)),
    v2.RandomResizedCrop(size=image_size, antialias=True)
    ])

cwd = os.getcwd()

train_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/training'), transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataloader.HarveyData(os.path.join(cwd, 'dataset/testing'), transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DeepLabV3(num_input_channels, num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 20

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
        outputs = model(image)
        
        mask_resize = Compose([v2.Resize((28, 28), antialias=True)])
        mask = mask_resize(mask)
        
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
            
            outputs = model(image)
            
            mask_resize = Compose([v2.Resize((28, 28), antialias=True)])
            mask = mask_resize(mask)
        
            loss = criterion(outputs, mask)
            total_loss += loss.item()
            
            softmax = nn.Softmax(dim=1)
            predicted = torch.argmax(softmax(model(image)), axis=1, keepdim=True)
            predicted_images.append(predicted.cpu().numpy())
            
            correct_predictions += (predicted == mask).sum()
            total_pixels += torch.numel(predicted)
            dice_score += (2 * (predicted * mask).sum()) / ((predicted + mask).sum() + 1e-8)
    
    accuracy = correct_predictions / total_pixels * 100
    average_loss = total_loss / len(test_dataloader)
    dice_score = dice_score / len(test_dataset)

    print('Accuracy: %.4f ---- Loss: %.4f ---- Dice: %.4f' % (accuracy, total_loss / test_dataset.__len__(), dice_score))
   
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

#Visualization       
fig, axs = plt.subplots(batch_size, 3, figsize=(32, 32))

for i in range(batch_size):
    # Plot the input image    
    image, mask = test_dataset.get_item_no_transforms(i)
    
    axs[i, 0].imshow(image.numpy()[3:6, :, :].T, aspect='equal')
    axs[i, 0].set_title("Post-Disaster Image")
    axs[i, 0].axis('off')

    # Plot the predicted image
    predicted_images_flat = [item for sublist in predicted_images for item in sublist]
    axs[i, 1].imshow(predicted_images_flat[i].T, cmap="viridis", aspect='equal')  # Adjust the colormap as needed
    axs[i, 1].set_title("Predicted Image")
    axs[i, 1].axis('off')

    # Plot the ground truth mask
    axs[i, 2].imshow(mask.numpy()[0].T, cmap="viridis", aspect='equal')  # Assuming the mask is a single-channel image
    axs[i, 2].set_title("Ground Truth Mask")
    axs[i, 2].axis('off')

plt.show()