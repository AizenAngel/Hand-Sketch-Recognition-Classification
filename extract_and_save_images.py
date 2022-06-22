import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import platform
import pathlib
import random
from tqdm import tqdm
import os

import cv2
from PIL import Image


def make_classes(img_location, save_path):
    if os.path.exists(save_path):
        return
    
    os.mkdir(save_path)
    
    for _, _, files in os.walk(img_location):
        files = list(filter(lambda x: '.INFO' not in x, files))
        for file in tqdm(files):
            filePath = os.path.join(img_location, file)
            classImages = np.load(filePath)
            firstNClasses = classImages[:5000]

            folderName = file.replace('quickdraw_full_numpy_bitmap_', '').replace('.npy', '')
            saveFolderName = os.path.join(save_path, folderName)

            os.mkdir(saveFolderName)
            
            index = 0
            for image in firstNClasses:
                save_image_path = os.path.join(saveFolderName, str(index)) + '.png'
                img_array = image.reshape(28, 28)
                plt.imsave(save_image_path, img_array)
                index += 1

# dirname = '/home/aizenangel/Desktop/ML_Projekat/Hand-Sketch-Recognition-Classification/sketch-recognition/downloads/'
# save_path = '/home/aizenangel/Desktop/ML_Projekat/Hand-Sketch-Recognition-Classification/sketch-classes'

# make_classes(dirname, save_path)