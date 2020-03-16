# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:08:31 2020

@author: btgl1e14
"""
# Monet or Manet?

# Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Iniitialising the CNN

classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu')) # N.B. TensorFlow backend requires input_shape arguments in reverse.

# Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classifier.add(Flatten())

# ANN
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to image dataset
from keras.preprocessing.image import ImageDataGenerator

# From keras documentation > image preprocessing:
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/train',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory('dataset/validation',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(train_generator,
                    steps_per_epoch=8266, # ie. total number in training set
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=2066)

test_generator = test_datagen.flow_from_directory('dataset/test',
                                                        target_size=(64, 64),
                                                        batch_size=37,
                                                        class_mode=None)

probabilities = classifier.predict_generator(test_generator, 1000)
