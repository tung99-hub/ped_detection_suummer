# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 00:41:33 2021

@author: Tung Nguyen Quang
"""
import numpy as np
from multiclass_logitboost import *
from sklearn.model_selection import StratifiedKFold
import argparse

def predict(dataset_name, n_nodes, method, test_iters, output_file):
    """
    Helper function to process the dataset into ready-made training and testing sets
    for use in the algortihms, also gathers the scores and prints them out.
    """
    dataset_path = "../processed_datasets/{}_{}_features.npy".format(dataset_name, method)
    if dataset_name == "TAU" or dataset_name == "MIO-TCD":
        with open(dataset_path, "rb") as file:
            data_X = np.load(file)
            data_y = np.load(file)
        
        if output_file == 'None':
            f = None
        else:
            f = open("../results/multiclass_logitboost_outputs.txt", "a")
        
        # Using 5-fold CV
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        fold = 0
        scores_by_folds = []
        
        for train_index, test_index in skf.split(data_X, data_y):
            fold += 1
            train_X = data_X[train_index]
            train_y = data_y[train_index]
            test_X = data_X[test_index]
            test_y = data_y[test_index]
            
            if method == "R":
                scores = multiclass_logitboost_cov(train_X, train_y, test_X, test_y, test_iters, n_nodes)
            elif method == "E":
                scores = multiclass_logitboost(train_X, train_y, test_X, test_y, test_iters, n_nodes)
            
            scores_by_folds.append(scores)
            
            for i in range(len(test_iters)):
                print('Accuracy on fold {} after {} iters ='.format(fold, test_iters[i]), scores[i], file=f)
            print(file=f)
        
        mean_scores = np.mean(scores_by_folds, axis=0)
        for i in range(len(mean_scores)):
            print('Mean accuracy on fold {} after {} iters ='.format(i, test_iters[i]), mean_scores[i], file=f)
        print('Overall mean accuracy using {} multiclass logitboost on {} with {} nodes ='.format(method, dataset_name, n_nodes), np.mean(mean_scores), file=f)
        print(file=f)
    
    elif dataset_name == "CIFAR":
        with open(dataset_path, "rb") as file:
                train_X = np.load(file)
                train_y = np.load(file)
                test_X = np.load(file)
                test_y = np.load(file)  
        
        if output_file == 'None':
            f = None
        else:
            f = open("../results/multiclass_logitboost_outputs.txt", "a")
            
        if method == "R":
            scores = multiclass_logitboost_cov(train_X, train_y, test_X, test_y, test_iters, n_nodes)
        elif method == "E":
            scores = multiclass_logitboost(train_X, train_y, test_X, test_y, test_iters, n_nodes)
        
        for i in range(len(test_iters)):
            print('Accuracy after {} iters using {} multiclass logitboost on CIFAR with {} nodes ='.format(test_iters[i], method, n_nodes), scores[i], file=f)
        print(file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name", help="Name of dataset, can be either CIFAR, TAU or MIO-TCD")
    parser.add_argument("-n_nodes", help="Number of terminal nodes in the stumps")
    parser.add_argument("-method", help="First letter of the name of the logitboost version, E(uclidean) or R(iemannian)")
    parser.add_argument(
            "-test_iters",
            nargs="*",  # expects ≥ 0 arguments
            type=int,
            default=[50, 100, 200],  # default list if no arg value
            help="Ascending list containing the iteration numbers to evaluate the accuracy at, for example [1,10,100] will \
                calculate the accuracy of the algorithm at iterations 1, 10 and 100. The number of iterations will \
                automatically be the biggest one in the list (in this case it is 100). Input the iters as 1 10 100, not [1, 10, 100]"
        )
    parser.add_argument("-output", help="Use None to print directly to terminal, or anything else to print to a designated file (this cannot be changed)")
    
    args = parser.parse_args()
    predict(args.dataset_name, int(args.n_nodes), args.method, args.test_iters, args.output)