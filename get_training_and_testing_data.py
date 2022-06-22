from cProfile import label
import os
import numpy as np
import PIL
from PIL import Image
import random
import numpy as np
from tqdm import tqdm

def get_images_and_labels(images_path, num_classes=50, res=28):
    X = []
    y = []
    classes = []
    index = 0
    for dirname in tqdm(os.listdir(images_path)):
        complete_class_path = os.path.join(images_path, dirname)
        for im_filename in os.listdir(complete_class_path):
            complete_im_filename = os.path.join(complete_class_path, im_filename)
            im_data = Image.open(complete_im_filename).convert('L')
            im_data = np.array(im_data).reshape(res, res, 1)
            X.append(im_data)
            y.append(index)
            classes.append(complete_class_path)
        index += 1
        if (index >= num_classes):
            break
    return np.array(X), np.array(y), np.array(classes)