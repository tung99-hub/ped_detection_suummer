# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:43:47 2021

@author: Tung
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class AdaBoost():
    """
    An AdaBoost classifier, which uses an ensemble of weak classifiers (called
    stumps) to make a majority vote on the label of a sample.                                                                    
    
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
    
    alpha_list: list
        List of alpha values from each iteration.
        
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
        super(AdaBoost, self).__init__()
        self.test_iters = test_iters
        self.n_leaves = n_leaves
        self.method = method
        self.random_state = random_state
        if method != 'E':
            self.mu_list = []
        self.alpha_list =[]
        self.clf_list = []
        self.n_iters = np.max(test_iters)
        self.predictions = []
        self.scores = []
        
    def fit(self, X, y):
        """
        Build an AdaBoost model from training data X and y.

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
        w = np.ones(N) * 1/N
        n_classes = int(np.max(y) + 1)
        self.n_classes = n_classes
        
        for m in range(self.n_iters):
            # Create a decision stump
            stump = DecisionTreeClassifier(max_leaf_nodes=self.n_leaves, 
                               random_state=self.random_state)
            
            if self.method == 'E':
                stump.fit(X, y, sample_weight=w)
                
                self.clf_list.append(stump)
                yhat = stump.predict(X)
            else:
                # Compute weighted mean of the points
                mu = mean_riemann(X, sample_weight=w)
                self.mu_list.append(mu)
                
                # Map data points to tangent space
                X_tspace = tangent_space(X, mu)
                
                # Fit the classifier
                stump.fit(X_tspace, y, sample_weight=w)
                
                self.clf_list.append(stump)
                yhat = stump.predict(X_tspace)
                
            # Compute error(m)
            err = np.sum(np.multiply(w, yhat != y)) / np.sum(w)
            
            # Compute alpha(m)
            alpha = np.log((1 - err) / err) + np.log(n_classes - 1)
            self.alpha_list.append(alpha)
            
            # Update weights
            w = w * np.exp(alpha * (yhat != y))
        
            # Re-normalize weights
            w /= np.linalg.norm(w)
            
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
        class_pred = np.zeros((X.shape[0], self.n_classes))
        
        for m in range(self.n_iters):
            if self.method == 'E':
                yhat = self.clf_list[m].predict(X)
            else:
                X_tspace = tangent_space(X, self.mu_list[m])
                yhat = self.clf_list[m].predict(X_tspace)
            
            for k in range(self.n_classes):
                class_pred[:, k] += self.alpha_list[m] * (yhat == k)            
            if m + 1 in self.test_iters:
                y_pred = np.argmax(class_pred, axis=1)
                self.predictions.append(y_pred)
                self.scores.append(accuracy_score(y_pred, y))
                    
        return self.predictions