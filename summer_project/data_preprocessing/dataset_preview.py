# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:00:02 2021

@author: subject F
"""
import numpy as np
from utils import compute_eight_dimensional_feature, vech
import torchvision
import matplotlib.pyplot as plt
import glob
from skimage import io

classes = []
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
images = []

for path in classes:
    image_paths = glob.glob(path)
    im = io.imread(image_paths[0])
    images.append(im)

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(4, 3, 1)
ax1.imshow(images[0])
ax1.set_title('Ambulance')

ax2 = fig.add_subplot(4, 3, 2)
ax2.imshow(images[1])
ax2.set_title('Bus')

ax3 = fig.add_subplot(4, 3, 3)
ax3.imshow(images[2])
ax3.set_title('Car')

ax4 = fig.add_subplot(4, 3, 4)
ax4.imshow(images[3])
ax4.set_title('Cart')

ax5 = fig.add_subplot(4, 3, 5)
ax5.imshow(images[4])
ax5.set_title('Limousine')

ax6 = fig.add_subplot(4, 3, 6)
ax6.imshow(images[5])
ax6.set_title('Motorcycle')

ax7 = fig.add_subplot(4, 3, 7)
ax7.imshow(images[6])
ax7.set_title('Snowmobile')

ax8 = fig.add_subplot(4, 3, 8)
ax8.imshow(images[7])
ax8.set_title('Tank')

ax9 = fig.add_subplot(4, 3, 9)
ax9.imshow(images[8])
ax9.set_title('Taxi')

ax10 = fig.add_subplot(4, 3, 10)
ax10.imshow(images[9])
ax10.set_title('Truck')

ax11 = fig.add_subplot(4, 3, 11)
ax11.imshow(images[10])
ax11.set_title('Van')