# Multiclass classification using Boosting and SVM
This repository contains code of the following algorithms: Adaboost, Logitboost, kernel (RBF) SVM for the task of multiclass classification.

It includes both versions of the algorithms, one for the Riemannian case (where inputs are SPD matrices), and the other for the usual Euclidean vectors case.

In addition, there is also Adaboost for the binary classification case, but it has not been maintained for quite some time so it is not guaranteed to work.

For any related questions or comments, feel free to send an email to me at tung.3.nguyen@aalto.fi.

## Requirements
The code is made in Python 3.7 with an Anaconda distribution. Two external libraries have been used to support the main algorithms:

- The pyriemann package from https://github.com/alexandrebarachant/pyRiemann
- The psdlearning package from https://github.com/AmmarMian/Comparative_study_pedestrian_Eusipco

Additionally, to be able to run the code for preprocessing the CIFAR-10 dataset, it is recommended to check for the existence of the ```torchvision``` python package before executing. If the package does not exist, simply run ```pip install torchvision``` or ```conda install torchvision``` in the terminal.

## Datasets
Three datasets have been used for this repository:

- CIFAR-10, available at https://www.cs.toronto.edu/~kriz/cifar.html
- MIO-TCD, available at http://podoce.dinf.usherbrooke.ca/challenge/dataset/
- TAU-Vehicle, available at https://www.kaggle.com/c/vehicle/data

To use these datasets, download them and put them in the ```summer_project/Datasets``` folder (create one if it does not already exist).

For more information on the dataset and the preprocessing procedure, refer to the ```summary.pdf``` file in the main folder.

## Running the algorithms
- Download the datasets and put them in the Datasets folder as above.
- Run the relevant preprocessing scripts in the ```summer_project/data_preprocessing``` folder.
- Execute the necessary algorithms by running the python files in terminal, for example: ```python path/to/multiclass_adaboost_predict.py -dataset_name CIFAR -n_nodes 2 -method R -test_iters 50 100 200 -output None```
- The results can be found in the file specified in the ```results``` folder, or it will be shown in the terminal itself.

More information about the parameters can be found by using the ```-h``` feature on the algorithms. It is also possible to run the code directly from the files by using the ```predict()``` function with the necessary parameters in the algorithm code itself.

The details of the algorithms and testing procedure will not be mentioned here, but users can refer to the ```summary.pdf``` file in the main folder for a more detailed explanation.

## Running in Triton
Most of the algorithms here act on big datasets, and it usually takes days for a single run. Therefore, it is advised to leave the code running on Triton to not use up all resources on the local machine. An example script will be provided below, feel free to change or add parameters to better suit the job:

```
#!/bin/bash
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=tung.3.nguyen@aalto.fi

srun pip uninstall scikit-learn
srun pip install scikit-learn==0.22
srun python path/to/multiclass_adaboost_predict.py -dataset_name CIFAR -n_nodes 2 -method R -test_iters 50 100 200 -output None
```

It is also worth noting that a script to preprocess data in Triton can also be made in a similar way, in case that process takes too long.