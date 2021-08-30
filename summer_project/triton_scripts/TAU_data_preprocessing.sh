#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun python ../data_preprocessing/TAU_data_preprocessing.py