"""
Created on Wed Apr 25 14:14:25 2018

@author: Venkatesh Balaji
"""
# Part 1: Building CNN

# Step 1:
# Importing Libraries and packages
import keras
from keras.models import Sequential # Initialize sequence of NN layers
from keras.layers import Conv2D # Performs 2D convolution layers
from keras.layers import MaxPooling2D # Used to perform Max Pooling in 2D
from keras.layers import Flatten # Used to flattening
from keras.layers import Dense #Used to add fully connected layers

# Step2:
# Initialize NN

classifier= Sequential()

# Adding Convolution layer
classifier.add(Conv2D(filters=32,kernel_size=(3,3), data_format="channels_last", 
                             input_shape=(64,64,3), activation='relu'))

# Adding Pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding 2nd Convolution layer
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Adding 2nd Pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Adding Flattening Layer
classifier.add(Flatten())

# Adding Full Connection NN
classifier.add(Dense(units=128,activation='relu',))


classifier.add(Dense(units=1,activation='sigmoid',))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Image Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# 
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=(8000/128),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=(2000/128))


import numpy as np
from keras.preprocessing import image

test_image=image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)

y_pred2 = classifier.predict(test_image)

training_set.class_indices