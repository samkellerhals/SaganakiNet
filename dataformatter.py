# -*- coding: utf-8 -*-

"""Data formatting utility script
Puts each image into a corresponding directory
named after all classes present in the dataset. 
"""

import os
import pandas as pd
import numpy as np
import random
from shutil import copyfile, move, rmtree

# get labels

labels = pd.read_csv('data/train_labels.csv')

# list of classes

classes = labels.label.unique()

# file names in training directory

training_file_names = os.listdir('data/train_set/train_set')

# define path to folder containing label subdirectories

training_path = 'data/train_set_struc/'
validation_path = 'data/valid_set_struc/'

# function to create subdirectory paths

def directory_generator(path):
    directories = [path + str(w) for w in classes]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# create directory paths for training and validation data

directory_generator(training_path)
directory_generator(validation_path)

# data copier function

def data_copier(file_names, path):
	for file in file_names:
	    # return row of match
	    row = labels[labels['img_name'].str.match(file)]
	    # corresponding label directory
	    directory = str(row.label.values[0]) + '/'
	    # generate target file path
	    file_path = path + directory + file
	    # copy file inside correct directory
	    copyfile('data/train_set/train_set/' + file, file_path)

# copy training files into respective label folder

data_copier(training_file_names, training_path)

# create validation set filename list (80/20 split)

validation_samples = []

for folder in os.listdir(training_path):
    folder_path = training_path + folder
    files = os.listdir(folder_path)
    sample = random.sample(files, round(0.2 * len(files)))
    validation_samples.append(sample)

validation_file_names = []

for sublist in validation_samples:
	for image in sublist:
		validation_file_names.append(image)

# subtract validation_file_names from training_file_names

for file_string in validation_file_names:
	if file_string in training_file_names:
		training_file_names.remove(file_string)

# delete entire directory

rmtree(training_path, ignore_errors=True)

# create final data folders
directory_generator(training_path)
data_copier(training_file_names, training_path)
data_copier(validation_file_names, validation_path)