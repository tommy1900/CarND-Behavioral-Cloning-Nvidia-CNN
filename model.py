## Load Tools 
import math
import numpy as np
from PIL import Image         
import cv2
import os
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import getcwd
import csv
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda 
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
###--------------------------------------------------

## Data Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for image_path, measurement in batch_samples:
                center_image = cv2.imread(image_path) 
                # cvt the image into RGB
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                images.append(center_image)
                measurements.append(measurement)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

## Preprocessing the data
def getData():
    """
     Make the data set better by useing the left and right camera as well, add it into the image set and measurement set
    """
    csv_path = './data/driving_log.csv'
    csv_path_mydata = './data/Mydata/driving_log.csv'
    
    lines = []
    lines_Mydata = []
    image_paths = []
    measurements = []
    # Use the sample data first
    with open(csv_path, newline='') as csvfile:
        reader  = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)

        for line in reader:
            lines.append(line)

    # Get all camera data 
    correction = 0.25 
    for line in lines[1:]:
        # Center image path and angle
        # skip if car not moving  
        if float(line[6]) < 0.1 :
            continue
        image_paths.append('./data/' + line[0])
        measurements.append(float(line[3]))
        # Left image path and angle
        image_paths.append('./data/' + line[1])
        measurements.append(float(line[3]) + correction)
        # Right image path and angle
        image_paths.append('./data/' + line[2])
        measurements.append(float(line[3]) - correction)
    # Now add in my own test data    
   # with open(csv_path_mydata, newline='') as csvfile_mydata:
   #     reader  = csv.reader(csvfile_mydata, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)

   #     for line_Mydata in reader:
   #         lines_Mydata.append(line_Mydata)

    # Get all camera data 
   # correction = 0.25  
   # for line_Mydata in lines_Mydata[1:]:
        # Center image path and angle
        # skip if car not moving  
   #     if float(line_Mydata[6]) < 0.1 :
   #         continue
        #print('./data/' + line_Mydata[0].split('/')[-3]+'/'+line_Mydata[0].split('/')[-2]+'/'+line_Mydata[0].split('/')[-1])
   #     image_paths.append('./data/' + line_Mydata[0].split('/')[-3]+'/'+line_Mydata[0].split('/')[-2]+'/'+line_Mydata[0].split('/')[-1])
   #     measurements.append(float(line[3]))
        # Left image path and angle
   #     image_paths.append('./data/' + line_Mydata[1].split('/')[-3]+'/'+line_Mydata[1].split('/')[-2]+'/'+line_Mydata[1].split('/')[-1])
   #     measurements.append(float(line[3]) + correction)
        # Right image path and angle
   #     image_paths.append('./data/' + line_Mydata[2].split('/')[-3]+'/'+line_Mydata[2].split('/')[-2]+'/'+line_Mydata[2].split('/')[-1])
   #     measurements.append(float(line[3]) - correction)
    
    image_paths = np.array(image_paths)
    measurements = np.array(measurements)
    return (image_paths,measurements)

# Call the generate data function
image_paths, measurements = getData()
# split into train/test sets
samples = list(zip(image_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

## ```````````````
## Model structure
##    (NVIDIA)
## ```````````````

model = Sequential()
# Normalization with preprocessed image(No cropping needed here)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20), (0,0))))
# Add three 5x5 convolution layers 
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = "relu"))
# Dropout
#model.add(Dropout(0.50))

# Add two 3x3 convolution layers  
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))

# Add a flatten layer
model.add(Flatten())
#model.add(Dropout(.2))

# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, activation = "relu"))
# Dropout
#model.add(Dropout(0.50))

model.add(Dense(50, activation = "relu"))
model.add(Dense(10))

# Add a fully connected output layer
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

## Post processing 
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1, verbose=1)

## Save Model
model.save('model_test4.h5')

## Visualization
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

# plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
            #plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
