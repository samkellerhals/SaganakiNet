# -*- coding: utf-8 -*-

"""
Convolutional Neural Network Ensemble using skip connection, inception and squeeze excitation models.
"""

# Import libraries
import torch
import numpy as np
from random import randint, uniform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import csv
import os
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import pretrainedmodels
from collections import OrderedDict
from PIL import Image
import helpers

# set device
device = torch.device(helpers.PROCESSOR)

# setup training dataset
training_dataset = datasets.ImageFolder(helpers.train_dir)

# Generate predictions for all models
for path in os.listdir('bin/'): 
	if 'ion' in path:
		csv_name = 'data/' + path.replace('.pt','') + '.csv'
		model = helpers.setup_device('bin/' + path, device)
		model.class_to_idx = training_dataset.class_to_idx
		print('inception')
		helpers.make_predictions(helpers.test_dir, csv_name, model, 'inception')
		print(csv_name + ' generated.')
	else:
		csv_name = 'data/' + path.replace('.pt','') + '.csv'
		model = helpers.setup_device('bin/' + path, device)
		model.class_to_idx = training_dataset.class_to_idx
		print('other')
		helpers.make_predictions(helpers.test_dir, csv_name, model, 'other')
		print(csv_name + ' generated.')
	
# Read all model CSVs

vgg16_preds = helpers.read_model_csv('vgg16')
resnet152_x3_preds = helpers.read_model_csv('resnet152_x3')
seresnext101_preds = helpers.read_model_csv('seresnext101')
resnet152_x2_preds = helpers.read_model_csv('resnet152_x2')
xception_preds = helpers.read_model_csv('xception')
vgg19_preds = helpers.read_model_csv('vgg19')
inceptionv4_preds = helpers.read_model_csv('inceptionv4')
seresnext50_preds = helpers.read_model_csv('seresnext50')
inceptionresnet_preds = helpers.read_model_csv('inceptionresnet')

# Ensembling #

# # Dataframe storing all model results
soft = pd.concat([
	vgg16_preds, 
	resnet152_x3_preds, 
	seresnext101_preds,
	resnet152_x2_preds,
	seresnext50_preds, 
	inceptionresnet_preds, 
	inceptionv4_preds, 
	vgg19_preds,
	xception_preds], axis=1)


# store probability values for all models
probs = pd.DataFrame(soft[[
	'vgg16_prob', 
	'resnet152_x3_prob', 
	'seresnext101_prob',
	'resnet152_x2_prob',
	'seresnext50_prob', 
	'inceptionresnet_prob', 
	'inceptionv4_prob', 
	'vgg19_prob',
	'xception_prob']]) 

# output model_name (column) with highest probability
probs = pd.DataFrame(probs.idxmax(axis=1))
probs.rename(columns = {0 : 'max_class'}, inplace=True)

# add highest probability model name back into dataframe
soft = pd.concat([probs, soft], axis=1)
soft.head()

# new column with highest certainty class
def max_ensemble(row):
   if row['max_class'] == 'vgg16_prob':
      return row['vgg16_label']
   if row['max_class'] == 'resnet152_x3_prob':
     return row['resnet152_x3_label']
   if row['max_class'] == 'seresnext101_prob':
      return row['seresnext101_label']
   if row['max_class'] == 'resnet152_x2_prob':
     return row['resnet152_x2_label']
   if row['max_class'] == 'seresnext50_prob':
     return row['seresnext50_label']
   if row['max_class'] == 'inceptionresnet_prob':
     return row['inceptionresnet_label']
   if row['max_class'] == 'inceptionv4_prob':
     return row['inceptionv4_label']
   if row['max_class'] == 'vgg19_prob':
     return row['vgg19_label']
   if row['max_class'] == 'xception_prob':
     return row['xception_label']

# extract highest certainty classes and make final prediction
soft['max_ensemble_label'] = soft.apply(max_ensemble, axis=1)
ensemble = soft[['img_name','max_ensemble_label']]
prediction = ensemble.iloc[:,8:10]
prediction = prediction.rename(columns={'max_ensemble_label':'label'})
prediction.to_csv('9_model_ensemble.csv')
