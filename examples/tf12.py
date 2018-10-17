#!/usr/bin/python3
from __future__ import print_function

import collections
import io
import os, signal
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

import random

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3




def main():
  print("hello world!")
  
  base_dir = './cats_and_dogs_filtered'
  train_dir = os.path.join(base_dir, 'train')
  validation_dir = os.path.join(base_dir, 'validation')

  # Directory with our training cat pictures
  train_cats_dir = os.path.join(train_dir, 'cats')

  # Directory with our training dog pictures
  train_dogs_dir = os.path.join(train_dir, 'dogs')

  # Directory with our validation cat pictures
  validation_cats_dir = os.path.join(validation_dir, 'cats')
  
  # Directory with our validation dog pictures
  validation_dogs_dir = os.path.join(validation_dir, 'dogs')

  train_cat_fnames = os.listdir(train_cats_dir)
  print (train_cat_fnames[:10])

  train_dog_fnames = os.listdir(train_dogs_dir)
  train_dog_fnames.sort()
  print (train_dog_fnames[:10])

  print ('total training cat images:', len(os.listdir(train_cats_dir)))
  print ('total training dog images:', len(os.listdir(train_dogs_dir)))  
  print ('total validation cat images:', len(os.listdir(validation_cats_dir)))
  print ('total validation dog images:', len(os.listdir(validation_dogs_dir)))
 
  # Add our data-augmentation parameters to ImageDataGenerator
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

  # Note that the validation data should not be augmented!
  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
        train_dir, # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

  # Flow validation images in batches of 20 using test_datagen generator
  validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

 
  local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
  pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None)
  pre_trained_model.load_weights(local_weights_file)
 

  last_layer = pre_trained_model.get_layer('mixed7')
  print ('last layer output shape:', last_layer.output_shape)
  last_output = last_layer.output

  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(last_output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = layers.Dense(1024, activation='relu')(x)
  # Add a dropout rate of 0.2
  x = layers.Dropout(0.2)(x)
  # Add a final sigmoid layer for classification
  x = layers.Dense(1, activation='sigmoid')(x)

  # Configure and compile the model
  model = Model(pre_trained_model.input, x)

  unfreeze = False

  # Unfreeze all models after "mixed6"
  for layer in pre_trained_model.layers:
    if unfreeze:
      layer.trainable = True
    if layer.name == 'mixed6':
      unfreeze = True


  model.compile(loss='binary_crossentropy',
              optimizer=SGD(
                  lr=0.00001, 
                  momentum=0.9),
              metrics=['acc'])


  history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

  os.kill(os.getpid(), signal.SIGKILL)

##################################################

def showpictures(train_cats_dir,train_cat_fnames,train_dogs_dir,train_dog_fnames):
 # Parameters for our graph; we'll output images in a 4x4 configuration
  nrows = 6
  ncols = 6

  # Index for iterating over images
  pic_index = 0
  fig = plt.gcf()
  fig.set_size_inches(ncols * 6, nrows * 6)

  pic_index += 18
  next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[pic_index-18:pic_index]]
  next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[pic_index-18:pic_index]]

  for i, img_path in enumerate(next_cat_pix+next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

  plt.show()

##################################################

if __name__== "__main__":
  main()



