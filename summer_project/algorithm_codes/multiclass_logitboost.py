# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:04:44 2021

@author: subject F
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.tree import DecisionTreeRegressor
from scipy.special import softmax
from sklearn.metrics import accuracy_score

def _weights_and_response(y, prob, max_response=4.0):
    """Update the working weights and response for a boosting iteration."""
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

def multiclass_logitboost_cov(train_X, train_y, test_X, test_y, test_iters, n_leaves=2):
    """
    Fit and test a Logitboost model using covariance descriptors, using Algorithm 6
    in Friedman, Hastie, & Tibshirani (2000).
    The difference here is that features lie in Riemannian manifolds, so samples
    are SPD matrices instead of typical vectors.

    Parameters
    ----------
    train_X : Array of shape (n_train, n_features, n_features)
        Training covariance descriptors in the form of SPD matrices.
    train_y : Array of shape (n_train)
        Training labels.
    test_X : Array of shape (n_test, n_features, n_features)
        Testing samples.
    test_y : Array of shape (n_test)
        Testing labels.
    test_iters : list
        Which iterations to evaluate the accuracy on
    n_leaves : int, optional
        Number of terminal nodes for each stump. The default is 2.

    Returns
    -------
    results : list
        Accuracy of the model corresponding to each iteration specified in test_iters.
    """
    N = train_X.shape[0]
    n_classes = np.max(train_y) + 1
    n_iters = np.max(test_iters)
    clf_list = []
    mu_list = []
    results = []
    
    # Initialize with uniform class probabilities
    p = np.full(shape=(N, n_classes), fill_value=(1. / n_classes))
    
    F = np.zeros((N, n_classes))
    
    y = np.eye(n_classes)[train_y]
    
    for m in range(n_iters):
        clf_list.append([])
        mu_list.append([])
        f = []
        for j in range(n_classes):
            # Compute weights and responses
            w, z = _weights_and_response(y[:, j], p[:, j])
            
            # Mapping the data to tangent space of the Riemannian mean
            mu = mean_riemann(train_X, sample_weight=w)
            mu_list[m].append(mu)
            X_tspace = tangent_space(train_X, mu)
            
            # Fit a classifier
            stump = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=0)
            stump.fit(X_tspace, z, sample_weight=w)
            clf_list[m].append(stump)
            
            if m < n_iters - 1:
                f.append(stump.predict(X_tspace))
                
        if m < n_iters - 1:
            # Update F and p
            f = np.asarray(f).T
            f -= f.mean(axis=1, keepdims=True)
            f *= ((n_classes - 1) / n_classes)

            F += f
            p = softmax(F, axis=1)
            
        if m + 1 in test_iters:
            scores = [[estimator.predict(tangent_space(test_X, mu)) for estimator, mu in zip(estimators, means)]
                      for estimators, means in zip(clf_list, mu_list)]
            scores = np.sum(scores, axis=0).T
            y_pred =  scores.argmax(axis=1)
            results.append(accuracy_score(y_pred, test_y))
        
    return results

def multiclass_logitboost(train_X, train_y, test_X, test_y, test_iters, n_leaves=2):
    """
    Fit and test a Logitboost model using covariance descriptors, using Algorithm 6
    in Friedman, Hastie, & Tibshirani (2000).

    Parameters
    ----------
    train_X : Array of shape (n_train, n_features)
        Training vectors.
    train_y : Array of shape (n_train)
        Training labels.
    test_X : Array of shape (n_test, n_features)
        Testing samples.
    test_y : Array of shape (n_test)
        Testing labels.
    test_iters : list
        Which iterations to evaluate the accuracy on
    n_leaves : int, optional
        Number of terminal nodes for each stump. The default is 2.

    Returns
    -------
    results : list
        Accuracy of the model corresponding to each iteration specified in test_iters.
    """
    N = train_X.shape[0]
    n_classes = np.max(train_y) + 1
    n_iters = np.max(test_iters)
    clf_list = []
    results = []
    
    # Initialize with uniform class probabilities
    p = np.full(shape=(N, n_classes), fill_value=(1. / n_classes))
    
    F = np.zeros((N, n_classes))
    
    y = np.eye(n_classes)[train_y]
    
    for m in range(n_iters):
        clf_list.append([])
        f = []
        for j in range(n_classes):
            # Compute weights and responses
            w, z = _weights_and_response(y[:, j], p[:, j])
        
            # Fit a classifier
            stump = DecisionTreeRegressor(max_leaf_nodes=n_leaves)
            stump.fit(train_X, z, sample_weight=w)
            clf_list[m].append(stump)
            
            if m < n_iters - 1:
                f.append(stump.predict(train_X))
                
        if m < n_iters - 1:
            # Update F and p
            f = np.asarray(f).T
            f -= f.mean(axis=1, keepdims=True)
            f *= ((n_classes - 1) / n_classes)

            F += f
            p = softmax(F, axis=1)
            
        if m + 1 in test_iters:
            scores = [[estimator.predict(test_X) for estimator in estimators]
                      for estimators in clf_list]
            scores = np.sum(scores, axis=0).T
            y_pred = scores.argmax(axis=1)        
            results.append(accuracy_score(y_pred, test_y))
        
    return results