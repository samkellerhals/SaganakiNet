# -*- coding: utf-8 -*-

# libraries & paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import torch
from random import randint, uniform
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import pretrainedmodels

train_dir = 'training_data/train_set_struc'
valid_dir = 'valid_data/valid_set_struc'

# PARAMETERS #

# load pretrained model
model = models.vgg16(pretrained=True)

# predefined parameter ranges
zero_to_one = [0,1]
zero_five = [0,0.5]

### Network image input size
IMAGE_INPUT_SIZE = 224

### IMAGENET Normalisation Weights, Mean & STD DEV
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

### Data Augmentation & Transformation ###
ROTATION = randint(*[0, 360])
COLOR_JITTER = [uniform(*zero_to_one), uniform(*zero_to_one), 
                uniform(*zero_to_one), uniform(*zero_five)]
RANDOM_HOR_FLIP = [uniform(*zero_to_one)]
RANDOM_VER_FLIP = [uniform(*zero_to_one)]


# Data Transformations
training_transforms = transforms.Compose([transforms.RandomRotation(ROTATION),
                      transforms.RandomResizedCrop(IMAGE_INPUT_SIZE),
                      transforms.ColorJitter(brightness=COLOR_JITTER[0], 
                                             contrast=COLOR_JITTER[1], 
                                             saturation=COLOR_JITTER[2], 
                                             hue=COLOR_JITTER[3]),
                      transforms.RandomVerticalFlip(RANDOM_VER_FLIP[0]),
                      transforms.RandomHorizontalFlip(RANDOM_HOR_FLIP[0]),
                      transforms.ToTensor(),
                      transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)])
  

validation_transforms = transforms.Compose([transforms.Resize(IMAGE_INPUT_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGE_NET_MEAN, 
                                                                 IMAGE_NET_STD)])

# Load the datasets
training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Define DataLoaders
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

# put device into GPU mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Freeze pretrained model parameters
for parameter in model.parameters():
    parameter.requires_grad = False


from collections import OrderedDict

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 80)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

# Function for the validation pass

def validation(model, validateloader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy

# Loss function and gradient descent
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train the classifier

def train_classifier():
  
        epochs = 30
        steps = 0
        print_every = 50

        model.to('cuda')

        for e in range(epochs):
        
            model.train()
    
            running_loss = 0
    
            for images, labels in iter(train_loader):
        
                steps += 1
        
                images, labels = images.to('cuda'), labels.to('cuda')
        
                optimizer.zero_grad()
        
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if steps % print_every == 0:
                
                    model.eval()
                
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        validation_loss, accuracy = validation(model, validate_loader, criterion)
            
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
            
                    running_loss = 0
                    model.train()
                    
train_classifier()

# save the model after training

torch.save(model, 'vgg16_pretrained.pt')