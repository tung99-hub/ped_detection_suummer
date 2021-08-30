#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python ../multiclass_logitboost_predict.py ../processed_datasets/miotcd_euclidean_features.npy 7 200 2 Euclidean MIO-TCD