# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:14:42 2021

@author: Tung Nguyen Quang
"""
import numpy as np
from pyriemann.utils.distance import distance_logeuclid
from pyriemann.utils.mean import mean_logeuclid
from pyriemann.utils.base import logm
from psdlearning.utils.algebra import vec
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import argparse

def _compute_gamma_auto(X_train):
    """
    Compute gamma based on equation 12 in Mian, Raninen & Ollila (2020).

    Parameters
    ----------
    X_train : array of shape (n_samples, n_features)
        Training matrix

    Returns
    -------
    gamma : int
        Optimal gamma value for the kernel SVM

    """
    sigma = 0
    mu = mean_logeuclid(X_train)
    for matrix in X_train:
        sigma += np.power(distance_logeuclid(matrix, mu), 2)
    return len(X_train) / sigma

def multiclass_kernel_SVM(train_X, train_y, test_X, test_y, method):
    """
    Fit and train a kernel SVM model according to section III.B. of Mian, Raninen &
    Ollila (2020).
    This function covers both the Euclidean and Riemannian version.

    Parameters
    ----------
    train_X : Array of shape (n_train, n_features, n_features) or (n_train, n_features)
        Training samples.
    train_y : Array of shape (n_train)
        Training labels.
    test_X : Array of shape (n_test, n_features, n_features) or (n_test, n_features)
        Testing samples.
    test_y : Array of shape (n_test)
        Testing labels.
    method : str, has to be either 'R' or 'E'
        First letter of the method to be used, E(uclidean) or R(iemannian).

    Returns
    -------
    score : float
        Accuracy of the model on the test data.

    """
    if method == 'R':
        gamma = _compute_gamma_auto(train_X)
    
        # Take the logarithm of the SPD matrices
        logm_X_train = np.empty((train_X.shape[0], train_X.shape[1]**2))
        for i in range(train_X.shape[0]):
            logm_X_train[i] = vec(logm(train_X[i]))
        logm_X_test = np.empty((test_X.shape[0], test_X.shape[1]**2))
        for i in range(test_X.shape[0]):
            logm_X_test[i] = vec(logm(test_X[i]))
            
        # Compute pairwise distances between the training set with itself, as
        # well between the train and test set
        pairwise_distances = cdist(logm_X_train, logm_X_train, 'sqeuclidean')
        pairwise_distances_test = cdist(logm_X_test, logm_X_train, 'sqeuclidean')
        
        # Kernel function for the above pairwise distances
        gram_matrix = np.exp(-gamma*pairwise_distances)
        gram_matrix_test = np.exp(-gamma*pairwise_distances_test)
        
        # Training and testing a kernel SVM
        clf = SVC(kernel='precomputed', gamma=gamma, random_state=0, break_ties=True, class_weight='balanced')        
        clf.fit(gram_matrix, train_y)            
        y_pred = clf.predict(gram_matrix_test)        
        score = accuracy_score(y_pred, test_y)
        
    elif method == 'E':
        # Kernel function (a bit different in the Euclidean case)
        gram_train = np.dot(train_X, train_X.T)
        gram_test = np.dot(test_X, train_X.T)
        
        # Training and testing a kernel SVM
        clf = SVC(kernel='precomputed', random_state=0, break_ties=True, class_weight='balanced')
        clf.fit(gram_train, train_y)
        y_pred = clf.predict(gram_test)
        score = accuracy_score(y_pred, test_y)
    
    return score

def predict(dataset_name, method, output_file):
    """
    Helper function to process the dataset into ready-made training and testing sets
    for use in the algortihms, also gathers the scores and prints them out.
    """
    dataset_path = "../processed_datasets/{}_{}_features.npy".format(dataset_name, method)
    if output_file == 'None':
        file = None
    else:
        file = open('../results/multiclass_kernel_SVM_outputs.txt', 'a')
        
    print("Using " + method + " multiclass kernel SVM on " + dataset_name + " dataset", file=file)    
    # Different datasets require different strategies
    if dataset_name == 'CIFAR':
        #Dataset already contains separate training and testing sets
        with open(dataset_path, 'rb') as f:
            train_X = np.load(f)
            train_y = np.load(f)
            test_X = np.load(f)
            test_y = np.load(f)
        
        print('Score = ', multiclass_kernel_SVM(train_X, train_y, test_X, test_y, method), file=file)
        print(file=file)
    
    elif dataset_name in ['MIO-TCD', 'TAU']:
        with open(dataset_path, 'rb') as f:
            data_X = np.load(f)
            data_y = np.load(f)
        
        # Use 5-fold CV with these datasets
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        fold = 0
        scores = []
        
        for train_index, test_index in skf.split(data_X, data_y):
            fold += 1
            train_X = data_X[train_index]
            train_y = data_y[train_index]
            test_X = data_X[test_index]
            test_y = data_y[test_index]
            
            score = multiclass_kernel_SVM(train_X, train_y, test_X, test_y, method)
            print('Accuracy on fold ' + str(fold) + ' = ', score, file=file)
            print(file=file)
            scores.append(score)
        
        print('Average score across ' + str(n_splits) + ' folds = ', np.mean(scores), file=file)
        print(file=file)                  
                          
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name", help="Name of the dataset, can be either CIFAR, TAU or MIO-TCD")
    parser.add_argument("-method", help="First letter of the name of the logitboost version, E(uclidean) or R(iemannian)")
    parser.add_argument("-output", help="Use None to print directly to terminal, or anything else to print to a designated file (this cannot be changed)")

    args = parser.parse_args()
    predict(args.dataset_name, args.method, args.output_file)