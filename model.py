import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import platform
import pathlib
import random
from tensorflow import keras
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

def create_model(image_shape, num_classes = 50):
    model = tf.keras.models.Sequential()
    l2_reg = l2()
    
    model.add(tf.keras.layers.Convolution2D(
        input_shape=image_shape,
        kernel_size=5,
        filters=32,
        padding='same',
        activation=tf.keras.activations.relu,
        kernel_regularizer=l2_reg
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    ))

    model.add(tf.keras.layers.Convolution2D(
        kernel_size=3,
        filters=32,
        padding='same',
        activation=tf.keras.activations.relu,
        kernel_regularizer=l2_reg
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    ))

    model.add(tf.keras.layers.Convolution2D(
        kernel_size=3,
        filters=64,
        padding='same',
        activation=tf.keras.activations.relu,
        kernel_regularizer=l2_reg
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu,
        kernel_regularizer=l2_reg
    ))

    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))
    
    return model