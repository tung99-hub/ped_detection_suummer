# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:03:26 2021

@author: subject F
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.tree import DecisionTreeClassifier

def binary_adaboost_fit(X, y, M):
    N = y.shape[0]
    w = np.ones(N) * 1/N
    mu_list = []
    err_list = []
    alpha_list = []
    clf_list = []
    
    for m in range(M):
        # Compute weighted mean of the points
        mu = mean_riemann(X, sample_weight=w)
        mu_list.append(mu)
        
        # Map data points to tangent space
        X_tspace = tangent_space(X, mu)
        
        # Fit a classifier
        stump = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
        stump.fit(X_tspace, y, sample_weight=w)
        clf_list.append(stump)
        yhat = stump.predict(X_tspace)
        
        # Compute error(m)
        err = np.sum(np.multiply(w, yhat != y)) / np.sum(w)
        err_list.append(err)
        
        # Compute alpha(m)
        alpha = np.log((1 - err) / err)
        alpha_list.append(alpha)
        
        # Update weights
        w = w * np.exp(alpha * (yhat != y))
        
        # Re-normalize weights
        w /= np.linalg.norm(w)
               
    return alpha_list, clf_list, mu_list

def binary_adaboost_predict(X, alpha_list, clf_list, mu_list):
    N = X.shape[0]
    class_pred = np.zeros((N, 2))
    M = len(clf_list)
    
    # For each class
    for k in range(2):
        # For each training iteration
        for m in range(M):
            X_tspace = tangent_space(X, mu_list[m])
            yhat = clf_list[m].predict(X_tspace)
            class_pred[:, k] += alpha_list[m] * (yhat == k)
            
    return np.argmax(class_pred, axis=1)