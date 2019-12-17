# -*- coding: utf-8 -*-

# libraries & paths
import numpy as np
import pandas as pd
from random import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import pretrainedmodels

train_dir = 'training_data/train_set_struc'
valid_dir = 'valid_data/valid_set_struc'

# Apply transformations

training_transforms = transforms.Compose(
	[transforms.Resize(360),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(30),
	transforms.ToTensor(),
	transforms.Normalize(
		[0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose(
	[transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
		[0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])])

# Load the datasets

training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)

validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Define the dataloaders

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True)

validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

# put device into GPU mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and change last linear layer

model_name = 'resnet152'
model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
num_ftrs = model.last_linear.in_features
model.fc = nn.Linear(num_ftrs, 80)
model = model.to(device)


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

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.fc.parameters())), lr=0.001, momentum=0.9)

# Train the classifier

def train_classifier():
  
        epochs = 20
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

torch.save(model, 'resnet152_x2_pretrained.pt')