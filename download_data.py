from re import L
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

def download_data(save_path):
    if os.path.exists(save_path):
        return
    
    os.mkdir(save_path)
    tfds.load(name="quickdraw_bitmap", 
                with_info=True, 
                as_supervised=True,
                download = True,
                data_dir = save_path)