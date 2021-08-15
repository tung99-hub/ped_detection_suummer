# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import glob
from skimage import color, io
from utils import compute_eight_dimensional_feature

def get_cov_features(dataset_path, truth_value):
    image_paths = glob.glob(dataset_path)
    image_list = np.zeros((len(image_paths), 8, 8))
    for i in range(len(image_paths)):
        im = color.rgb2gray(io.imread(image_paths[i]))
        im_features = compute_eight_dimensional_feature(im)
        w, h, d = im_features.shape
        image_list[i, :, :] = np.cov(im_features.reshape(w*h, d).T)

    if truth_value:
        return image_list, np.ones(len(image_list))
    return image_list, np.zeros(len(image_list))
        
train_X_neg = '../Datasets/INRIAPerson/Train/neg/*.png'
train_X_pos = '../Datasets/INRIAPerson/Train/pos/*.png'
test_X_pos = '../Datasets/INRIAPerson/Test/pos/*.png'
test_X_neg = '../Datasets/INRIAPerson/Test/neg/*.png'

train_X_neg, train_y_neg = get_cov_features(train_X_neg, False)
train_X_pos, train_y_pos = get_cov_features(train_X_pos, True)
test_X_neg, test_y_neg = get_cov_features(test_X_neg, False)
test_X_pos, test_y_pos = get_cov_features(test_X_pos, True)

train_X = np.vstack((train_X_neg, train_X_pos))
train_y = np.hstack((train_y_neg, train_y_pos))
test_X = np.vstack((test_X_neg, test_X_pos))
test_y = np.hstack((test_y_neg, test_y_pos))

with open('../INRIA_R_features.npy', 'wb') as f:
    np.save(f, train_X)
    np.save(f, train_y)
    np.save(f, test_X)
    np.save(f, test_y)