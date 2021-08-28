#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip install scikit-learn==0.22
srun python ../algorithm_codes/multiclass_kernel_svm.py -method E -dataset_name MIO-TCD -output yes
srun python ../algorithm_codes/multiclass_kernel_svm.py -method R -dataset_name MIO-TCD -output yes