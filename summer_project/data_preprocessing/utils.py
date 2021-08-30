# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:39:36 2021

@author: subject F
"""
import numpy as np
from scipy import ndimage
from skimage import color, io
import glob
import logging
import time
from tqdm import tqdm

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

def get_features(dataset_path, num_regions):
    image_paths = glob.glob(dataset_path)
    N = len(image_paths) * num_regions
    cov_list = np.zeros((N, 8, 8))
    euc_list = np.zeros((N, 36))
    for i in range(len(image_paths)):
        im = color.rgb2gray(io.imread(image_paths[i]))
        w, h = im.shape
        sub_regions_list = generate_sub_regions_random(num_regions, w, h, 1.25, 1.25)
        for j in range(len(sub_regions_list)):
            x_j, y_j, w_j, h_j = sub_regions_list[j]
            sub_im = im[ y_j : y_j+h_j, x_j : x_j+w_j ]
            im_features = compute_eight_dimensional_feature(sub_im)
            w, h, d = im_features.shape
            cov_matrix = np.cov(im_features.reshape(w*h, d).T)
            cov_list[i*num_regions + j, :, :] = cov_matrix
            euc_list[i*num_regions + j, :] = vech(cov_matrix)

    return euc_list, cov_list

def process_classes_list(list_of_classes, num_regions):
    samples = []
    samples_cov = []
    labels = []
    
    for i in range(len(list_of_classes)):
        print('Processing class', i)
        data, data_cov = get_features(list_of_classes[i], num_regions)
        samples.append(data)
        samples_cov.append(data_cov)
        labels.append(np.ones(data.shape[0]) * i)
        
    return samples, samples_cov, labels

def compute_overlap_percent(region_1, region_2):
    """ Function to compute the overlap in percent between two regions.

        Usage: overlap_percent = compute_overlap_percent(region_1, region_2)
        Inputs:
            * region_1 = an list of ints corresponding to x_1, y_1, w_1, h_1
            * region_2 = an list of ints corresponding to x_2, y_2, w_2, h_2
        Outputs:
            * overlap_percent = a float corresponding to the overlap in percent.
    """
    x_1, y_1, w_1, h_1 = region_1
    x_2, y_2, w_2, h_2 = region_2

    if x_1 < x_2:
        w_j_tilde = w_1
    else:
        w_j_tilde = w_2

    if y_1 < y_2:
        h_j_hat = h_1
    else:
        h_j_hat = h_2

    if (np.abs(x_1-x_2) < w_j_tilde) and ( np.abs(y_1-y_2)<h_j_hat ):
        overlap_percent = 100 * (h_j_hat - np.abs(y_1-y_2)) * \
                 (w_j_tilde - np.abs(x_1-x_2)) / np.min( (h_1*w_1, h_2*w_2) )
    else:
        overlap_percent = 0

    return overlap_percent

def generate_sub_regions_random_with_overlap_constraint(N_R, h, w, n_h, n_w, overlap_threshold=75, timeout=30,
                                                    seed=None, progress=False):
    """ Function to obtain a uniformly random sampling for the sub_regions.

        Usage: sub_regions_list = generate_sub_regions_random_with_overlap_constraint(N_R, h, w, n_h,
                                                    n_w, overlap_threshold, timeout, seed, progress)
        Inputs:
            * N_R = an int corresponding to number of regions to generate.
            * h = an int corresponding to height of the image.
            * w = an int corresponding to width of the image.
            * n_w = an int so that the sub_regions are of width that is minimum \ceil{w/n_w}.
            * n_h = an int so that the sub_regions are of height that is minimum \ceil{h/n_h}.
            * overlap_threshold = a float corresponding to the maximum overlap in percent.
            * timeout = a float corresponding to the timeout limit in minutes of this function.
            * seed = an int or a rng which is the seed for rng so that it is reproducible.
            * progress = a boolean to show or not a progress bar for the generation.
        Outputs:
            * sub_regions_list = a list of ints [x_j, y_j, w_j, h_j] where:
                - (x_j, y_j) are the coordinates of the left corner of the region,
                - (w_j, h_j) are the width and heigth of the region.
    """

    if seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    sub_regions_list = []
    j = 0
    if progress:
        pbar = tqdm(total=N_R)
    t_beginning = time.time()
    timeout_seconds = timeout*60
    while j<N_R and (time.time()-t_beginning)<timeout_seconds:

        # randomly generate using uniform distribution a set of values
        x_j = rng.randint(0, w - int(np.ceil(w/n_w)))
        y_j = rng.randint(0, h - int(np.ceil(h/n_h)))
        w_j = rng.randint(int(np.ceil(w/n_w)), w - x_j)
        h_j = rng.randint(int(np.ceil(h/n_h)), h - y_j)

        # Checking overlap
        is_not_overlapping = True
        for region in sub_regions_list:
            if compute_overlap_percent( region, (x_j, y_j, w_j, h_j) ) > overlap_threshold:
                is_not_overlapping = False
                break
        if is_not_overlapping:
            sub_regions_list.append( (x_j, y_j, w_j, h_j) )
            j = j + 1
            if progress:
                pbar.update(1)
    if j < N_R:
        logging.warning("Timed out after %.2f minutes and %d sub_regions", (time.time()-t_beginning)/60, j)
        logging.warning("The remainder of generated sub-regions won't have the overlap constraint")
        for index in range(j, N_R):
            x_j = rng.randint(0, w - int(np.ceil(w/n_w)))
            y_j = rng.randint(0, h - int(np.ceil(h/n_h)))
            w_j = rng.randint(int(np.ceil(w/n_w)), w - x_j)
            h_j = rng.randint(int(np.ceil(h/n_h)), h - y_j)
            sub_regions_list.append( (x_j, y_j, w_j, h_j) )
            if progress:
                pbar.update(1)

    return sub_regions_list

def generate_sub_regions_random(N_R, h, w, n_h, n_w, seed=None, progress=False):
    """ Function to obtain a uniformly random sampling for the sub_regions.

        Usage: sub_regions_list = generate_sub_regions_random(N_R, h, w, n_h, n_w, seed, progress)
        Inputs:
            * N_R = an int corresponding to number of regions to generate.
            * h = an int corresponding to height of the image.
            * w = an int corresponding to width of the image.
            * n_w = an int so that the sub_regions are of width that is minimum \ceil{w/n_w}.
            * n_h = an int so that the sub_regions are of height that is minimum \ceil{h/n_h}.
            * seed = an int which is the seed for rng so that it is reproducible.
            * progress = a boolean to show or not a progress bar for the generation.
        Outputs:
            * sub_regions_list = a list of ints [x_j, y_j, w_j, h_j] where:
                - (x_j, y_j) are the coordinates of the left corner of the region
                - (w_j, h_j) are the width and heigth of the region
    """


    if seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    sub_regions_list = []
    if progress:
        pbar = tqdm(total=N_R)
    for j in range(N_R):
        x_j = rng.randint(0, w - int(np.ceil(w/n_w)))
        y_j = rng.randint(0, h - int(np.ceil(h/n_h)))
        w_j = rng.randint(int(np.ceil(w/n_w)), w - x_j)
        h_j = rng.randint(int(np.ceil(h/n_h)), h - y_j)
        sub_regions_list.append( (x_j, y_j, w_j, h_j) )
        if progress:
                pbar.update(1)
    return sub_regions_list