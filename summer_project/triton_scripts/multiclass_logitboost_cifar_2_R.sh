#!/bin/bash
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python ../multiclass_logitboost_predict.py ../processed_datasets/CIFAR_cov_features.npy 10 200 2 Riemannian CIFAR
