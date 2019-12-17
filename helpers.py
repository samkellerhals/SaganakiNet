# -*- coding: utf-8 -*-

"""
Helper functions and variables to make ensemble predictions. Specify what processor you are using below and how many images out of the test you would like to predict.
"""

import torch
from PIL import Image
import numpy as np
import os
import csv
import pandas as pd

# PARAMETERS #

PREDICT_ROWS = 5 # Set number of rows to predict 
PROCESSOR = 'cpu' # cpu for testing, gpu otherwise

# Data Paths
train_dir = 'training_data/train_set_struc'
valid_dir = 'valid_data/valid_set_struc'
test_dir = 'test_data/test_set/test_set/'

# Setup device
def setup_device(model_path, device):
    	model = torch.load(model_path, map_location=torch.device(PROCESSOR))
    	model.eval()
    	model.to(device)
    	return model

# Reading model CSVs
def read_model_csv(code):
	
	for csv in os.listdir('data/'):
		
		if code in csv:
			model_df = pd.read_csv('data/' + csv, names=["img_name", code + "_label", code + "_prob"])
			
			return model_df

# Predict and Image Processing Functions
def process_image(image_path, model):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    pil_image = Image.open(image_path)
    
    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    if model == 'inception':
    	dim = 399
    	mean = np.array([0.5, 0.5, 0.5])
    	std = np.array([0.5, 0.5, 0.5])
    elif model == 'other':
    	dim = 224
    	mean = np.array([0.485, 0.456, 0.406])
    	std = np.array([0.229, 0.224, 0.225])

    # Crop 
    left_margin = (pil_image.width-dim)/2
    bottom_margin = (pil_image.height-dim)/2
    right_margin = left_margin + dim
    top_margin = bottom_margin + dim
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    np_image = (np_image - mean) / std
    
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, model_type, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path, model_type)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    
    # Turn output into SOFTMAX probabilities
    probabilities = torch.exp(output)/torch.sum(torch.exp(output), dim=1)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    for i in range(len(top_indices)):
      if(top_indices[i] > 80):
        top_indices[i] = 0
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes

# Make model predictions
def make_predictions(test_dir, model_name, model, type):
	
	with open(model_name, mode='w') as preds:

		if type == 'other':
			for image in os.listdir(test_dir)[0:PREDICT_ROWS]: # TEST
				probs, classes = predict(test_dir + image, model, 'other')
				preds_writer = csv.writer(preds, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				preds_writer.writerow([image, classes[0], probs[0]])
		else:
			for image in os.listdir(test_dir)[0:PREDICT_ROWS]: # TEST
				probs, classes = predict(test_dir + image, model, 'inception')
				preds_writer = csv.writer(preds, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				preds_writer.writerow([image, classes[0], probs[0]])
