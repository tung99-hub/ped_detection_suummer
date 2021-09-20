# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 23:40:52 2021

@author: Tung
"""
import numpy as np
from pyriemann.utils.distance import distance_logeuclid
from pyriemann.utils.mean import mean_logeuclid
from pyriemann.utils.base import logm
from psdlearning.utils.algebra import vec
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def _compute_gamma_auto(X_train):
    """
    Compute gamma based on equation 12 in Mian, Raninen & Ollila (2020).

    Parameters
    ----------
    X_train : array of shape (n_samples, n_features, n_features)
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

class KernelSVM():
    """
    An SVM model that uses the Radial Basis Function (RBF) kernel to estimate  
    distances between data points.
    
    Parameters
    ----------
    method: "E" or "R", optional (default "E")
        Which version of AdaBoost to use. "E" denotes the typical Euclidean space
        of vectors, while "R" involves the Riemannian space of SPD matrices.
        
    random_state: int, optional (default None)
        Optional random seed for reproducibility.
        
    Attributes
    ----------
    clf: object
        The trained kernel SVM instance.
    
    scores: list
        Single element list containing the accuracy of clf on the test set.
        
    result: list
        Single element list containing the predicted labels of clf on the test set.
        
    logm_X_train: array of shape (n_samples, 8, 8)
        Log of the training matrix. Only useful in the Riemannian case.
        
    gamma:
        Learning rate of the kernel SVM. Only useful in the Riemannian case.
    """
    
    def __init__(self, method, random_state):
        super(KernelSVM, self).__init__()
        self.method = method
        self.random_state = random_state
        self.clf = None
        self.scores = []
        self.result = []
        self.logm_X_train = []
        if method != 'E':
            self.gamma = 0
    
    def fit(self, X, y):
        """
        Build a kernel SVM model from training data X and y.

        Parameters
        ----------
        X : array of shape (n_samples, n_features) or (n_samples, 8, 8)
            Training samples in Euclidean or Riemannian space.
        y : array of shape (n_samples)
            Training labels.

        Returns
        -------
        None.

        """
        if self.method == 'E':
            self.clf = SVC(kernel='rbf', gamma='scale', random_state=self.random_state, class_weight='balanced')
            self.clf.fit(X, y)
        else:
            gamma = _compute_gamma_auto(X)
            self.gamma = gamma
            
            # Take the logarithm of the SPD matrices
            logm_X = np.empty((X.shape[0], X.shape[1]**2))
            for i in range(X.shape[0]):
                logm_X[i] = vec(logm(X[i]))
            self.logm_X_train = logm_X
                
            # Compute pairwise distances between the training set with itself, as
            # well between the train and test set
            pairwise_distances = cdist(logm_X, logm_X, 'sqeuclidean')
            
            # Kernel function for the above pairwise distances
            gram_matrix = np.exp(-gamma*pairwise_distances)
            
            # Training and testing a kernel SVM
            self.clf = SVC(kernel='precomputed', gamma=gamma, random_state=self.random_state, class_weight='balanced')        
            self.clf.fit(gram_matrix, y)
            
    def predict(self, X, y):
        """
        Use the fitted model to predict the labels of testing samples, while also
        giving the accuracy score.

        Parameters
        ----------
        X : array of shape (n_samples, n_features) or (n_samples, 8, 8)
            Testing samples in Euclidean or Riemannian space.
        y : array of shape (n_samples)
            Testing labels.

        Returns
        -------
        self.result
            See class docstring for more info.

        """
        if self.method == 'E':
            y_pred = self.clf.predict(X)
            self.result = y_pred
            self.scores.append(accuracy_score(y_pred, y))
            return y_pred
        else:
            logm_X = np.empty((X.shape[0], X.shape[1]**2))
            for i in range(X.shape[0]):
                 logm_X[i] = vec(logm(X[i]))
            
            pairwise_distances = cdist(logm_X, self.logm_X_train, 'sqeuclidean')
            gram_matrix = np.exp(-self.gamma*pairwise_distances)
            
            y_pred = self.clf.predict(gram_matrix)
            self.result = y_pred
            self.scores.append(accuracy_score(y_pred, y))
            return y_pred