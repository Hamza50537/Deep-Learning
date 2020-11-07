#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop


# In[11]:


tf.__version__


# # Loading the Dataset

# In[12]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[13]:


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print(x_test.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')


# In[14]:


for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()


# # Data Preprocessing (Image Augmentation)
# * Here we will apply the transformations on the training set images only.
# * It's being done so to avoid the overfitting.
# * If we don't apply the transformtions than it will have a huge difference between the accuracy of training and test dataset.
# * We change the images dimensions,zoom in, zoom out e.t.c.

# In[15]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


# # Building the Model

# In[16]:


cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5) ,activation='relu', padding='Same', input_shape=[32,32,3]))  #Conv2d Layer
cnn.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu')) #2nd Conv2d Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)) #1st Pooling Layer
cnn.add(Dropout(0.25))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5) ,padding='Same', activation='relu')) # 3rd Conv2d Layer
cnn.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same', activation='relu'))  # 4th Conv2d Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)) #2nd Pooling Layer
cnn.add(Dropout(0.25))

cnn.add(tf.keras.layers.Flatten()) #Flattening Layer
cnn.add(tf.keras.layers.Dense(units=256,activation="relu",name="layer1")) #1st Dense here we will use the larger no of neurons(128).
cnn.add(Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=10,activation="softmax",name="layer2")) #Output Layer 1 neuron cat vs dog.


# # Training the Model

# In[18]:


cnn.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


# In[19]:


datagen.fit(x_train)
cnn.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=15)
#cnn.fit(train_generator,validation_data=validation_generator,epochs=25) #batches were already defined in preprocessing step


# * It's taking time so i have just train it for 15 epochs with more epochs it will have better accuarcy on training data.

# # Testing 

# In[20]:


# Score trained model.
scores = cnn.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# make prediction.
pred = cnn.predict(x_test)


# # Visualizing the Images

# In[21]:


for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i])
plt.show()

