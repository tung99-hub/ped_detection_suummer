# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:21:05 2021

@author: subject F
"""
import os
import numpy as np
from utils import compute_eight_dimensional_feature

def pgm_to_covariance_features(dataset_path):
    """Return a raster of integers from a PGM as a list of lists."""
    image_paths = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            image_paths.append(os.path.join(root,file))
    image_list = np.zeros((len(image_paths), 8, 8))
    labels_list = np.zeros(len(image_paths))
    label_loc = len(dataset_path) + 1
    for i in range(len(image_paths)):
        print(i)
        # Check folder name to determine pos/neg sample
        if image_paths[i][label_loc] == 'p':
            labels_list[i] = 1
        pgmf = open(image_paths[i], 'rb')
        assert pgmf.readline().decode() == 'P5\n'
        (width, height) = [int(i) for i in pgmf.readline().decode().split()]
        depth = int(pgmf.readline().decode())
        assert depth <= 255
    
        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(pgmf.read(1)))
            raster.append(row)
        image_features = compute_eight_dimensional_feature(np.array(raster))
        w, h, d = image_features.shape
        image_list[i, :, :] = np.cov(image_features.reshape(w*h, d).T)
    return image_list, labels_list

train_1_path = '../Datasets/DC-ped-dataset_base/1'
train_2_path = '../Datasets/DC-ped-dataset_base/2'
train_3_path = '../Datasets/DC-ped-dataset_base/1'
test_1_path = '../Datasets/DC-ped-dataset_base/T1'
test_2_path = '../Datasets/DC-ped-dataset_base/T2'

train_X_1, train_y_1 = pgm_to_covariance_features(train_1_path)
train_X_2, train_y_2 = pgm_to_covariance_features(train_2_path)
train_X_3, train_y_3 = pgm_to_covariance_features(train_3_path)
test_X_1, test_y_1 = pgm_to_covariance_features(test_1_path)
test_X_2, test_y_2 = pgm_to_covariance_features(test_2_path)

with open('../processed_datasets/DC_R_features.npy', 'wb') as f:
    np.save(f, train_X_1)
    np.save(f, train_y_1)
    np.save(f, train_X_2)
    np.save(f, train_y_2)
    np.save(f, train_X_3)
    np.save(f, train_y_3)
    np.save(f, test_X_1)
    np.save(f, test_y_1)
    np.save(f, test_X_2)
    np.save(f, test_y_2)