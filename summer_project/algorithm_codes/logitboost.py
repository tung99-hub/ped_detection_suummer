# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:15:49 2021

@author: Tung
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.tree import DecisionTreeRegressor
from scipy.special import softmax
from sklearn.metrics import accuracy_score

def _weights_and_response(y, prob, max_response=4.0):
    """Update the working weights and response for a LogitBoost iteration."""
    # Samples with very certain probabilities (close to 0 or 1) are weighted
    # less than samples with probabilities closer to 1/2. This is to
    # encourage the higher-uncertainty samples to contribute more during
    # model training
    sample_weight = prob * (1. - prob)

    # The smallest representable 64 bit floating point positive number eps such that
    # 1.0 + eps != 1.0
    _MACHINE_EPSILON = np.finfo(np.float64).eps
    
    # Don't allow sample weights to be too close to zero for numerical
    # stability (cf. p. 353 in Friedman, Hastie, & Tibshirani (2000)).
    sample_weight = np.maximum(sample_weight, 2. * _MACHINE_EPSILON)

    # Compute the regression response z = (y - p) / (p * (1 - p))
    with np.errstate(divide='ignore', over='ignore'):
        z = np.where(y, 1. / prob, -1. / (1. - prob))

    # Very negative and very positive values of z are clipped for numerical
    # stability (cf. p. 352 in Friedman, Hastie, & Tibshirani (2000)).
    z = np.clip(z, a_min=-max_response, a_max=max_response)

    return sample_weight, z

class LogitBoost(): 
    """
    A LogitBoost classifier, which uses an ensemble of weak classifiers (called
    stumps) to minimize the logistic likelihood loss.                                                                  
    
    Parameters
    ----------
    test_iters: list
        The iterations at which to estimate the accuracy of the model.
    
    n_leaves: int
        Number of terminal nodes on each stump.
        
    method: "E" or "R", optional (default "E")
        Which version of AdaBoost to use. "E" denotes the typical Euclidean space
        of vectors, while "R" involves the Riemannian space of SPD matrices.
        
    random_state: int, optional (default None)
        Whether or not to use a random seed for reproducibility.
        
    Attributes
    ----------
    mu_list: list
        List of weighted means after each iteration. Only useful in Riemannian case.
        
    clf_list: list
        List of stumps from each iteration.
        
    n_iters: int
        Number of iterations to run, inferred from test_iters as its maximum number.
        
    predictions: list
        List of prediction labels from each iteration specified in test_iters.
        
    scores: list
        List of each accuracy score from each iteration specified in test_iters.
        
    n_classes: int
        Number of classes in the dataset, inferred as the max number in labels
        list + 1.    
    """
    
    def __init__(self, test_iters, n_leaves, method='E', random_state=None):
        super(LogitBoost, self).__init__()
        self.test_iters = test_iters
        self.n_leaves = n_leaves
        self.method = method
        self.random_state = random_state
        if method != 'E':
            self.mu_list = []
        self.clf_list = []
        self.n_iters = np.max(test_iters)
        self.predictions = []
        self.scores = []
        
    
    def fit(self, X, y):
        """
        Build an LogitBoost model from training data X and y.

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
        N = y.shape[0]
        n_classes = int(np.max(y) + 1)
        self.n_classes = n_classes
        
        # Initialize with uniform class probabilities
        p = np.full(shape=(N, n_classes), fill_value=(1. / n_classes))        
        F = np.zeros((N, n_classes))        
        y = np.eye(n_classes)[y.astype(np.int)]
        
        for m in range(self.n_iters):
            self.clf_list.append([])
            if self.method != 'E':
                self.mu_list.append([])
            f = []
            
            for j in range(n_classes):
                # Compute weights and responses
                w, z = _weights_and_response(y[:, j], p[:, j])
                stump = DecisionTreeRegressor(max_leaf_nodes=self.n_leaves, \
                                              random_state=self.random_state)    
                if self.method == 'E':
                    stump.fit(X, z, sample_weight=w)
                    self.clf_list[m].append(stump)
                    if m < self.n_iters - 1:
                        f.append(stump.predict(X))
                else:
                    # Mapping the data to tangent space of the Riemannian mean
                    mu = mean_riemann(X, sample_weight=w)
                    self.mu_list[m].append(mu)
                    X_tspace = tangent_space(X, mu)
                    
                    # Fit the classifier
                    stump.fit(X_tspace, z, sample_weight=w)
                    self.clf_list[m].append(stump)
                    if m < self.n_iters - 1:
                        f.append(stump.predict(X_tspace))
                        
            if m < self.n_iters - 1:
                # Update F and p
                f = np.asarray(f).T
                f -= f.mean(axis=1, keepdims=True)
                f *= ((n_classes - 1) / n_classes)
    
                F += f
                p = softmax(F, axis=1)
                
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
        self.predictions
            See class docstring for more info.

        """
        for m in self.test_iters:
            if self.method == 'E':
                scores = [ [estimator.predict(X) for estimator in estimators]
                      for estimators in self.clf_list[:m] ]
            else:
                scores = [ [estimator.predict(tangent_space(X, mu)) 
                            for estimator, mu in zip(estimators, means)] 
                          for estimators, means in zip(self.clf_list[:m], self.mu_list[:m])]
            scores = np.sum(scores, axis=0).T
            y_pred = scores.argmax(axis=1)
            self.predictions.append(y_pred)
            self.scores.append(accuracy_score(y_pred, y))
            
        return self.predictions