#!/bin/bash
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python ../multiclass_adaboost_predict.py ../processed_datasets/miotcd_euclidean_features.npy 7 200 2 Euclidean MIO-TCD