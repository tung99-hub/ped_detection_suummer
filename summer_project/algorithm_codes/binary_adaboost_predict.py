# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:53:59 2021

@author: subject F
"""
import numpy as np
from sklearn.metrics import accuracy_score
from binary_adaboost import binary_adaboost_fit, binary_adaboost_predict
from sklearn.model_selection import StratifiedKFold

def predict(dataset, output_file=None):
    if dataset == 'DC':
        print('Using binary Adaboost on DaimerChrysler dataset', file=output_file)
        # 3 training set, 2 testing set
        with open('DC_cov_features.npy', 'rb') as f:
            train_X_1 = np.load(f)
            train_y_1 = np.load(f)
            train_X_2 = np.load(f)
            train_y_2 = np.load(f)
            train_X_3 = np.load(f)
            train_y_3 = np.load(f)
            test_X_1 = np.load(f)
            test_y_1 = np.load(f)
            test_X_2 = np.load(f)
            test_y_2 = np.load(f)
        
        # Combine 2 training batches to create a trainset, then test the classifier on both testset
        alpha_list1, clf_list1, mu_list1 = binary_adaboost_fit(np.vstack((train_X_2, train_X_3)), np.hstack((train_y_2, train_y_3)), 100)
        y_pred11 = binary_adaboost_predict(test_X_1, alpha_list1, clf_list1, mu_list1)
        y_pred12 = binary_adaboost_predict(test_X_2, alpha_list1, clf_list1, mu_list1)
        score11 = accuracy_score(test_y_1, y_pred11)
        score12 = accuracy_score(test_y_2, y_pred12)
        print('Accuracy of fold 1 validation on test set 1 =', score11, file=output_file)
        print('Accuracy of fold 1 validation on test set 2 =', score12, file=output_file)
        
        alpha_list2, clf_list2, mu_list2 = binary_adaboost_fit(np.vstack((train_X_1, train_X_3)), np.hstack((train_y_1, train_y_3)), 100)
        y_pred21 = binary_adaboost_predict(test_X_1, alpha_list2, clf_list2, mu_list2)
        y_pred22 = binary_adaboost_predict(test_X_2, alpha_list2, clf_list2, mu_list2)
        score21 = accuracy_score(test_y_1, y_pred21)
        score22 = accuracy_score(test_y_2, y_pred22)
        print('Accuracy of fold 2 validation on test set 1 =', score21, file=output_file)
        print('Accuracy of fold 2 validation on test set 2 =', score22, file=output_file)
        
        alpha_list3, clf_list3, mu_list3 = binary_adaboost_fit(np.vstack((train_X_1, train_X_2)), np.hstack((train_y_1, train_y_2)), 100)
        y_pred31 = binary_adaboost_predict(test_X_1, alpha_list3, clf_list3, mu_list3)
        y_pred32 = binary_adaboost_predict(test_X_2, alpha_list3, clf_list3, mu_list3)
        score31 = accuracy_score(test_y_1, y_pred31)
        score32 = accuracy_score(test_y_2, y_pred32)
        print('Accuracy of fold 3 validation on test set 1 =', score31, file=output_file)
        print('Accuracy of fold 3 validation on test set 2 =', score32, file=output_file)
        
    elif dataset == 'INRIA':
        print('Using binary Adaboost on INRIA dataset', file=output_file)
        # Using 4-fold CV
        with open('INRIA_cov_features.npy', 'rb') as f:
            train_X = np.load(f)
            train_y = np.load(f)
            test_X = np.load(f)
            test_y = np.load(f)
            
        skf = StratifiedKFold(n_splits=4, shuffle=True)
        fold = 0
        scores = np.zeros(4)
        for train_index, test_index in skf.split(train_X, train_y):
            fold += 1
            X_train = train_X[train_index]
            y_train = train_y[train_index]
            
            alpha_list, clf_list, mu_list = binary_adaboost_fit(X_train, y_train, 100)
            y_pred = binary_adaboost_predict(test_X, alpha_list, clf_list, mu_list)
            score = accuracy_score(test_y, y_pred)
            scores[fold-1] = score
            print('Accuracy of fold ' + str(fold) + ' on test set =', score, file=output_file)
        
        print('Mean score across 4 folds =', np.mean(scores), file=output_file)

with open("binary_adaboost_predict_outputs.txt", "w") as output_file:
    predict('INRIA', output_file)
    predict('DC', output_file)