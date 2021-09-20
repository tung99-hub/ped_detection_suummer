# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:53:25 2021

@author: Tung
"""
import numpy as np
from adaboost import AdaBoost
from logitboost import LogitBoost
from kernel_svm import KernelSVM
from sklearn.model_selection import StratifiedKFold
import argparse

def get_clf(algo, test_iters, n_leaves, method, random_state):
    '''Function to generate a model according to the algorithm name.'''
    if algo.lower() == 'adaboost':
        clf = AdaBoost(test_iters, n_leaves, method, random_state)
    elif algo.lower() == 'logitboost':
        clf = LogitBoost(test_iters, n_leaves, method, random_state)
    elif algo.lower() == 'kernelsvm':
        clf = KernelSVM(method)
    else:
        raise ValueError("Algorithm {} not found!".format(algo))
    return clf

def predict(dataset_name, algo, n_leaves=0, method='E', test_iters=[0], output_file='None', random_state=None):
    """
    Helper function to process the dataset into ready-made training and testing sets
    for use in the algortihms, also gathers the scores and prints them out.
    
    This function only supports a handful of datasets that have been heavily preprocessed,
    however the algorithms can still be used on every other datasets with a little coding.
    """
    # Construct path to data inside this repo
    dataset_path = "../processed_datasets/{}_{}_features.npy".format(dataset_name, method)
    
    if dataset_name == "TAU" or dataset_name == "MIO-TCD":
        with open(dataset_path, "rb") as file:
            data_X = np.load(file)
            data_y = np.load(file)
        
        # Choose to print to an external file or not
        if output_file == 'None':
            f = None
        else:
            f = open("../results/multiclass_adaboost_outputs.txt", "a")
        
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

            clf = get_clf(algo, test_iters, n_leaves, method, random_state)
            clf.fit(train_X, train_y)
            y_pred = clf.predict(test_X, test_y)
            
            scores_by_folds.append(clf.scores)
            
            for i in range(len(test_iters)):
                print('Accuracy on fold {} after {} iters =' \
                      .format(fold, test_iters[i]), clf.scores[i], file=f)
            print(file=f)            
            
        # The print statements might be a bit inaccurate for kernel SVM runs
        mean_scores = np.mean(scores_by_folds, axis=0)
        for i in range(len(mean_scores)):
            print('Mean accuracy after {} iters ='.format(test_iters[i]), mean_scores[i], file=f)
        
        print('Overall mean accuracy using {} {} on {} with {} nodes =' \
              .format(method, algo, dataset_name, n_leaves), np.mean(mean_scores), file=f)
        print(file=f)
    # Similar to the above        
    elif dataset_name == "CIFAR":
        with open(dataset_path, "rb") as file:
                train_X = np.load(file)
                train_y = np.load(file)
                test_X = np.load(file)
                test_y = np.load(file)  
        
        if output_file == 'None':
            f = None
        else:
            f = open("../results/multiclass_adaboost_outputs.txt", "a")
            
        clf = get_clf(algo, test_iters, n_leaves, method, random_state)
        clf.fit(train_X, train_y)
        result = clf.predict(test_X, test_y)
        
        for i in range(len(test_iters)):
            print('Accuracy after {} iters using {} {} on CIFAR with {} nodes =' \
                  .format(test_iters[i], algo, method, n_leaves), clf.scores[i], file=f)
        print(file=f)
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name", help="Name of dataset, can be either CIFAR, TAU or MIO-TCD")
    parser.add_argument("-algo", help="Name of algorithm to execute, currently available \
                        are AdaBoost, LogitBoost and KernelSVM")
    parser.add_argument("-n_leaves", help="Number of terminal nodes in the stumps")
    parser.add_argument("-method", help="First letter of the name of the Adaboost version, \
                        E(uclidean) or R(iemannian)")
    parser.add_argument(
            "-test_iters",
            nargs="*",
            type=int,
            default=[50, 100, 200],  # default list if no arg value
            help="Ascending list containing the iteration numbers to evaluate \
                the accuracy at, for example [1,10,100] will calculate the accuracy \
                of the algorithm at iterations 1, 10 and 100. The number of iterations \
                will automatically be the biggest one in the list (in this case it is 100). \
                Input the iters as 1 10 100, not [1, 10, 100]"
        )
    parser.add_argument("-output", help="Use None to print directly to terminal, \
                        or anything else to print to a designated file (this cannot be changed)")
    parser.add_argument("-random_state", help="Random seed for reproducibility. \
                        Disabled by default, use an int to enable.")
    
    args = parser.parse_args()
    
    predict(dataset_name=args.dataset_name, 
            algo=args.algo, 
            test_iters=args.test_iters, 
            n_leaves=int(args.n_leaves), 
            method=args.method, 
            output_file=args.output, 
            random_state=int(args.random_state))
