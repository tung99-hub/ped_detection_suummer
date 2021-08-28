#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=quang.t.nguyen@aalto.fi

srun pip install scikit-learn==0.22
srun python ../algorithm_codes/multiclass_adaboost_predict.py -dataset_name CIFAR -n_nodes 2 -test_iters 50 100 200 -output yes -method E 
srun python ../algorithm_codes/multiclass_adaboost_predict.py -dataset_name CIFAR -n_nodes 8 -test_iters 50 100 200 -output yes -method E 
srun python ../algorithm_codes/multiclass_adaboost_predict.py -dataset_name CIFAR -n_nodes 2 -test_iters 50 100 200 -output yes -method R
srun python ../algorithm_codes/multiclass_adaboost_predict.py -dataset_name CIFAR -n_nodes 8 -test_iters 50 100 200 -output yes -method R
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name CIFAR -n_nodes 2 -test_iters 50 100 200 -output yes -method E 
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name CIFAR -n_nodes 2 -test_iters 50 100 200 -output yes -method R
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name CIFAR -n_nodes 8 -test_iters 50 100 200 -output yes -method E 
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name CIFAR -n_nodes 8 -test_iters 50 100 200 -output yes -method R
srun python ../algorithm_codes/multiclass_kernel_svm.py -method E -dataset_name CIFAR -output yes
srun python ../algorithm_codes/multiclass_kernel_svm.py -method R -dataset_name CIFAR -output yes