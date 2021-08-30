#!/bin/bash
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun python ../data_preprocessing/CIFAR_data_preprocessing.py