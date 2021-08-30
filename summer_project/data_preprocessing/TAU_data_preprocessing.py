# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:07:06 2021

@author: subject F
"""
import numpy as np
from utils import process_classes_list

classes = []
# Get directory of the 11 classes
classes.append('../Datasets/TAU_vehicles/train/Ambulance/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Bus/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Car/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Cart/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Limousine/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Motorcycle/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Snowmobile/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Tank/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Taxi/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Truck/*.jpg')
classes.append('../Datasets/TAU_vehicles/train/Van/*.jpg')
num_regions = 5

samples, samples_cov, labels = process_classes_list(classes, num_regions)

data_X = np.vstack(samples)
data_X_cov = np.vstack(samples_cov)
data_y = np.hstack(labels).astype(np.uint8)

with open("../processed_datasets/TAU_R_features.npy", "wb") as f:
    np.save(f, data_X_cov)
    np.save(f, data_y)
    
with open("../processed_datasets/TAU_E_features.npy", "wb") as f:
    np.save(f, data_X)
    np.save(f, data_y) 