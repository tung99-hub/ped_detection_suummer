# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:39:36 2021

@author: subject F
"""
import numpy as np
from scipy import ndimage
from skimage import color, io
import glob

def compute_eight_dimensional_feature(image):
    """ A function to compute the 8-dimensional features for an image as used in:
        O. Tuzel, F. Porikli and P. Meer,
        "Pedestrian Detection via Classification on Riemannian Manifolds",
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 30, no. 10, pp. 1713-1727, Oct. 2008.
        doi: 10.1109/TPAMI.2008.75,

        at eq. (11) p. 1716.

        Usage: image_features = compute_eight_dimensional_feature(image)
        Inputs:
            * image = a numpy array of shape (h, w) corresponding to the image.
        Outputs:
            * image_features = a numpy array of shape (h, w, 8) corresponding
                                to the tensor of image features."""

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x,y)
    Ix = ndimage.sobel(image,axis=1,mode='constant')
    Ixx = ndimage.sobel(Ix,axis=1,mode='constant')
    Iy = ndimage.sobel(image,axis=0,mode='constant')
    Iyy = ndimage.sobel(Iy,axis=0,mode='constant')
    I_abs = np.hypot(np.abs(Ix), np.abs(Iy))
    A = np.arctan2(np.abs(Iy), np.abs(Ix))

    return np.dstack([X, Y, np.abs(Ix), np.abs(Iy),
                        I_abs, np.abs(Ixx), np.abs(Iyy), A])

def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))

def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols

def get_features(dataset_path):
    image_paths = glob.glob(dataset_path)
    N = len(image_paths)
    cov_list = np.zeros((N, 8, 8))
    euc_list = np.zeros((N, 36))
    for i in range(N):
        im = color.rgb2gray(io.imread(image_paths[i]))
        im_features = compute_eight_dimensional_feature(im)
        w, h, d = im_features.shape
        cov_matrix = np.cov(im_features.reshape(w*h, d).T)
        cov_list[i, :, :] = cov_matrix
        euc_list[i, :] = vech(cov_matrix)

    return euc_list, cov_list

def process_classes_list(list_of_classes):
    samples = []
    samples_cov = []
    labels = []
    
    for i in range(len(list_of_classes)):
        print('Processing class ', i)
        data, data_cov = get_features(list_of_classes[i])
        samples.append(data)
        samples_cov.append(data_cov)
        labels.append(np.ones(data.shape[0]) * i)
        
    return samples, samples_cov, labels