#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# In[7]:


tf.__version__


# # Data Preprocessing (Image Augmentation)
# * Here we will apply the transformations on the training set images only.
# * It's being done so to avoid the overfitting.
# * If we don't apply the transformtions than it will have a huge difference between the accuracy of training and test dataset.
# * We change the images dimensions,zoom in, zoom out e.t.c.
# 

# ## Preprocessing The Training Data

# In[8]:


train_datagen = ImageDataGenerator(
        rescale=1./255,  #feature scaling each pixel(all pixel values between 0 and 1)
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(80, 80),
        batch_size=32,
        class_mode='binary') #cat vs dog only 2 options


# ## Preprocessing The Training Data
# * Here we have applied the only rescale option just to have image pixels range between o to 1 just like train dataset.
# * Here we have not applied the the other options that we had done for the test dataset.

# In[9]:


test_datagen = ImageDataGenerator(rescale=1./255) 
validation_generator = test_datagen.flow_from_directory(  
        'dataset/test_set',
        target_size=(80, 80),
        batch_size=32,
        class_mode='binary')


# # Building the Model

# In[11]:


cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3 ,activation='relu', input_shape=[80,80,3]))  #Conv2d Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)) #1st Pooling Layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3 ,activation='relu')) # 2nd Conv2d Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)) #2nd Pooling Layer
cnn.add(tf.keras.layers.Flatten()) #Flattening Layer
cnn.add(tf.keras.layers.Dense(units=128,activation="relu",name="layer1")) #1st Dense here we will use the larger no of neurons(128).
cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid",name="layer2")) #Output Layer 1 neuron cat vs dog.


# # Training the CNN

# In[13]:


cnn.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(train_generator,validation_data=validation_generator,epochs=25) #batches were already defined in preprocessing step


# # Making Predictions

# In[14]:


import numpy as np
from tensorflow.keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                          target_size=(80,80))
test_image=image.img_to_array(test_image) #Predict method expects it in 2d array
test_image=np.expand_dims(test_image,axis=0) #Dimension of the batch that we are adding will be first dimmension.
result=cnn.predict(test_image)
# 1 corresponds to dog and 0 corresponds to cat
train_generator.class_indices
if result[0][0]==1: #result[batch][pic no]
    prediction='dog'
else:
    prediction='cat'
    
print(prediction)

