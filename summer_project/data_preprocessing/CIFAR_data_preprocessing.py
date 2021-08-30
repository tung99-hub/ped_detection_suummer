# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:45:25 2021

@author: subject F
"""
import numpy as np
from utils import *
import torchvision

data_dir = '../Datasets'
trainset = torchvision.datasets.CIFAR10(root=data_dir)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False)
num_regions = 2
subregion_ratio = 1.25
train_sample_size = len(trainset) * num_regions
test_sample_size = len(testset) * num_regions

# Convert RGB image to grayscale
train_data = np.dot(np.array(trainset.data)[...,:3], [0.299 , 0.587, 0.114])
train_y = np.zeros(train_sample_size)
test_data = np.dot(np.array(testset.data)[...,:3], [0.299 , 0.587, 0.114])
test_y = np.zeros(test_sample_size)

train_X = np.zeros((train_sample_size, 8, 8))
test_X = np.zeros((test_sample_size, 8, 8))
train_X_euc = np.zeros((train_sample_size, 36))
test_X_euc = np.zeros((test_sample_size, 36))

# Process training samples
sub_regions = []
for i in range(len(trainset)):
    print(i)
    im  = train_data[i, :, :]
    sub_regions_list = generate_sub_regions_random(num_regions, 32, 32, subregion_ratio, subregion_ratio)
    sub_regions.append(sub_regions_list)
    label = int(trainset.targets[i])
    for j in range(len(sub_regions_list)):
        x_j, y_j, w_j, h_j = sub_regions_list[j]    
        sub_im = im[ y_j : y_j+h_j, x_j : x_j+w_j ]
        im_features = compute_eight_dimensional_feature(sub_im)
        w, h, d = im_features.shape
        cov_matrix = np.cov(im_features.reshape(w*h, d).T)
        train_X[i*num_regions + j, :, :] = cov_matrix
        train_X_euc[i*num_regions + j, :] = vech(cov_matrix)
        train_y[i*num_regions + j] = label
    
# Process testing samples
for i in range(len(testset)):
    print(i)
    im = test_data[i, :, :]
    sub_regions_list = generate_sub_regions_random(num_regions, 32, 32, num_regions, subregion_ratio)
    label = int(testset.targets[i])
    for j in range(len(sub_regions_list)):
        x_j, y_j, w_j, h_j = sub_regions_list[j]    
        sub_im = im[ y_j : y_j+h_j, x_j : x_j+w_j ]
        im_features = compute_eight_dimensional_feature(sub_im)
        w, h, d = im_features.shape
        cov_matrix = np.cov(im_features.reshape(w*h, d).T)
        test_X[i*num_regions + j, :, :] = cov_matrix
        test_X_euc[i*num_regions + j, :] = vech(cov_matrix)
        test_y[i*num_regions + j] = label
    
with open("../processed_datasets/CIFAR_R_features.npy", "wb") as f:
    np.save(f, train_X)
    np.save(f, train_y)
    np.save(f, test_X)
    np.save(f, test_y)
    
with open("../processed_datasets/CIFAR_E_features.npy", "wb") as f:
    np.save(f, train_X_euc)
    np.save(f, train_y)
    np.save(f, test_X_euc)
    np.save(f, test_y)