#!/bin/bash
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python ../multiclass_adaboost_predict.py ../processed_datasets/CIFAR_euclidean_features.npy 10 200 8 Euclidean CIFAR