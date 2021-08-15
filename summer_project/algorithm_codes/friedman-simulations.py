# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 15:35:46 2021

@author: subject F
"""
import numpy as np
from multiclass_logitboost import multiclass_logitboost_fit_euclidean, multiclass_logitboost_predict_euclidean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

with open('Datasets/glass-data/glass.data', 'rb') as f:
    lines = f.readlines()
    
data = []
for line in lines:
    data.append(line.decode().split(",")[1:])
    
data_X = []
data_y = []
for sample in data:
    data_X.append(sample[:-1])
    data_y.append(sample[-1])

data_X = np.array(data_X, dtype=np.float32)
data_y = np.array(data_y, dtype=np.int) - 1

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
fold = 0
scores = np.zeros(n_splits)
errors = np.zeros(n_splits)
for train_index, test_index in skf.split(data_X, data_y):
    fold += 1
    X_train = data_X[train_index]
    y_train = data_y[train_index]
    X_test = data_X[test_index]
    y_test = data_y[test_index]
    
    clf_list = multiclass_logitboost_fit_euclidean(X_train, y_train, 200, 7, 8)
    y_pred = multiclass_logitboost_predict_euclidean(X_test, clf_list, 200)
    score = accuracy_score(y_test, y_pred)
    test_error = 1 - score
    scores[fold-1] = score
    errors[fold-1] = test_error
    #print('Accuracy of fold ' + str(fold) + ' on test set =', score)
    print('Test error rate on fold ' + str(fold) + ' = ', test_error)

print('Mean test error rate across ' + str(n_splits) + ' folds =', np.mean(errors))