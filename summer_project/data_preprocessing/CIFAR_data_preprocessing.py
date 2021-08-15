# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:45:25 2021

@author: subject F
"""
import numpy as np
from utils import compute_eight_dimensional_feature, vech
import torchvision

data_dir = '../Datasets'
trainset = torchvision.datasets.CIFAR10(root=data_dir)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False)
train_sample_size = len(trainset)
test_sample_size = len(testset)

# Convert RGB image to grayscale
train_data = np.dot(np.array(trainset.data)[...,:3], [0.299 , 0.587, 0.114])
train_y = np.array(trainset.targets)[:train_sample_size]
test_data = np.dot(np.array(testset.data)[...,:3], [0.299 , 0.587, 0.114])
test_y = np.array(testset.targets)[:test_sample_size]

train_X = np.zeros((train_sample_size, 8, 8))
test_X = np.zeros((test_sample_size, 8, 8))
train_X_euc = np.zeros((train_sample_size, 36))
test_X_euc = np.zeros((test_sample_size, 36))

# Process training samples
for i in range(train_sample_size):
    im  = train_data[i, :, :]
    im_features = compute_eight_dimensional_feature(im)
    w, h, d = im_features.shape
    cov_matrix = np.cov(im_features.reshape(w*h, d).T)
    train_X[i, :, :] = cov_matrix
    train_X_euc[i, :] = vech(cov_matrix)
    
# Process testing samples
for i in range(test_sample_size):
    im = test_data[i, :, :]
    im_features = compute_eight_dimensional_feature(im)
    w, h, d = im_features.shape
    cov_matrix = np.cov(im_features.reshape(w*h, d).T)
    test_X[i, :, :] = cov_matrix
    test_X_euc[i, :] = vech(cov_matrix)
    
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