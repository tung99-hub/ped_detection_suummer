# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:18:55 2021

@author: subject F
"""
import numpy as np
import glob
from utils import compute_eight_dimensional_feature, vech, get_features, \
                    process_classes_list

classes = []
classes.append('../Datasets/MIO-TCD-Classification/train/articulated_truck/*.jpg')
classes.append("../Datasets/MIO-TCD-Classification/train/bus/*.jpg")
classes.append("../Datasets/MIO-TCD-Classification/train/motorcycle/*.jpg")
classes.append("../Datasets/MIO-TCD-Classification/train/non-motorized_vehicle/*.jpg")
classes.append("../Datasets/MIO-TCD-Classification/train/single_unit_truck/*.jpg")
classes.append("../Datasets/MIO-TCD-Classification/train/work_van/*.jpg")

samples, samples_cov, labels = process_classes_list(classes)

data_X = np.vstack(samples)
data_X_cov = np.vstack(samples_cov)
data_y = np.hstack(labels).astype(np.uint8)

with open("../processed_datasets/MIO-TCD_R_features.npy", "wb") as f:
    np.save(f, data_X_cov)
    np.save(f, data_y)
    
with open("../processed_datasets/MIO-TCD_E_features.npy", "wb") as f:
    np.save(f, data_X)
    np.save(f, data_y)  