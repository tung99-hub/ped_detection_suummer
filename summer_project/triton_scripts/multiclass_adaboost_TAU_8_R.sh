#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python ../multiclass_adaboost_predict.py ../processed_datasets/TAU_cov_features.npy 17 200 8 Riemannian TAU