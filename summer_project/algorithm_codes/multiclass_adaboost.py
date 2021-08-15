# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:49:31 2021

@author: subject F
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def multiclass_adaboost_cov(train_X, train_y, test_X, test_y, test_iters, n_leaves=2):
    """
    Perform multiclass Adaboost fitting and testing according to Algorithm 2 in 
    Zhu, Zou, Rosset & Hastie (2009), but the features now lie in the Riemannian manifolds.

    Parameters
    ----------
    train_X : Array of shape (n_train, n_features, n_features)
        Training samples, an array of covariance descriptors in the form of SPD matrices.
    train_y : Array of shape (n_train)
        List of class labels according to the training samples.
    test_X : Array of shape (n_test, n_features, n_features)
        Testing samples in the form of SPD matrices.
    test_y : Array of shape (n_test)
        Labels for the testing samples.
    test_iters : List
        Which iterations to evaluate the accuracy on.
    n_leaves : int, optional
        Number of terminal nodes for each stump. The default is 2.

    Returns
    -------
    scores : List
        Contains the accuracies corresponding to each iteration specified in test_iters.

    """
    N = train_y.shape[0]
    w = np.ones(N) * 1/N
    n_classes = np.max(train_y) + 1
    n_iters = np.max(test_iters)
    mu_list = []
    err_list = []
    alpha_list = []
    clf_list = []
    
    for m in range(n_iters):
        # Compute weighted mean of the points
        mu = mean_riemann(train_X, sample_weight=w)
        mu_list.append(mu)
        
        # Map data points to tangent space
        X_tspace = tangent_space(train_X, mu)
        
        # Fit a classifier
        stump = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=0)
        stump.fit(X_tspace, train_y, sample_weight=w)
        clf_list.append(stump)
        yhat = stump.predict(X_tspace)
        
        # Compute error(m)
        err = np.sum(np.multiply(w, yhat != train_y)) / np.sum(w)
        err_list.append(err)
        
        # Compute alpha(m)
        alpha = np.log((1 - err) / err) + np.log(n_classes - 1)
        alpha_list.append(alpha)
        
        # Update weights
        w = w * np.exp(alpha * (yhat != train_y))
    
        # Re-normalize weights
        w /= np.linalg.norm(w)
        
    class_pred = np.zeros((test_y.shape[0], n_classes))
    scores = []
    
    # For each class
    for m in range(n_iters):
        X_tspace = tangent_space(test_X, mu_list[m])
        yhat = clf_list[m].predict(X_tspace)
        # For each training iteration
        for k in range(n_classes):
            class_pred[:, k] += alpha_list[m] * (yhat == k)            
        if m + 1 in test_iters:
            y_pred = np.argmax(class_pred, axis=1)
            scores.append(accuracy_score(y_pred, test_y))

    return scores

def multiclass_adaboost(train_X, train_y, test_X, test_y, test_iters, n_leaves=2):
    """
    Perform multiclass Adaboost fitting and testing according to Algorithm 2 in 
    Zhu, Zou, Rosset & Hastie (2009).

    Parameters
    ----------
    train_X : Array of shape (n_train, n_features)
        Training samples, an array of covriance matrices that have been flattened to vectors.
    train_y : Array of shape (n_train)
        List of class labels according to the training samples.
    test_X : Array of shape (n_test, n_features)
        Testing samples.
    test_y : Array of shape (n_test)
        Labels for the testing samples.
    test_iters : List
        Which iterations to evaluate the accuracy on.
    n_leaves : int, optional
        Number of terminal nodes for each stump. The default is 2.

    Returns
    -------
    scores : List
        Contains the accuracies corresponding to each iteration specified in test_iters.

    """
    N = train_y.shape[0]
    w = np.ones(N) * 1/N
    n_classes = np.max(train_y) + 1
    n_iters = np.max(test_iters)
    err_list = []
    alpha_list = []
    clf_list = []
    
    for m in range(n_iters):
        # Fit a classifier
        stump = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=0)
        stump.fit(train_X, train_y, sample_weight=w)
        clf_list.append(stump)
        yhat = stump.predict(train_X)
        
        # Compute error(m)
        err = np.sum(np.multiply(w, yhat != train_y)) / np.sum(w)
        err_list.append(err)
        
        # Compute alpha(m)
        alpha = np.log((1 - err) / err) + np.log(n_classes - 1)
        alpha_list.append(alpha)
        
        # Update weights
        w = w * np.exp(alpha * (yhat != train_y))
    
        # Re-normalize weights
        w /= np.linalg.norm(w)

    class_pred = np.zeros((test_y.shape[0], n_classes))
    scores = []
    
    # For each class
    for m in range(n_iters):
        yhat = clf_list[m].predict(test_X)
        # For each training iteration
        for k in range(n_classes):
            class_pred[:, k] += alpha_list[m] * (yhat == k)            
        if m + 1 in test_iters:
            y_pred = np.argmax(class_pred, axis=1)
            scores.append(accuracy_score(y_pred, test_y))

    return scores