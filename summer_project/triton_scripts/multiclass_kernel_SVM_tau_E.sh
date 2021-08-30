#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python ../multiclass_kernel_svm.py ../processed_datasets/TAU_euclidean_features.npy E TAU
