#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip install scikit-learn==0.22
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name MIO-TCD -n_nodes 2 -test_iters 50 100 200 -output yes -method E 
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name MIO-TCD -n_nodes 8 -test_iters 50 100 200 -output yes -method E 
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name MIO-TCD -n_nodes 2 -test_iters 50 100 200 -output yes -method R
srun python ../algorithm_codes/multiclass_logitboost_predict.py -dataset_name MIO-TCD -n_nodes 8 -test_iters 50 100 200 -output yes -method E 