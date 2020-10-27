#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


tf.__version__


# # Data Preprocessing

# In[7]:


dataset=pd.read_csv('Churn_Modelling.csv')
df=pd.DataFrame(dataset)
X=df.iloc[:,3:-1].values
y=df.iloc[:,-1].values


# In[12]:


X


# In[13]:


y


# ## Encoding Categorical Data 
# * Here in our problem Geography and Gender are categorical variable so we will apply the one hot encoding and label encoding particular feature.

# ### Encoding Gender Feature
# * Here we have applied the label encoder because there are only 2 types of gender male or female.

# In[32]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X[:,2])


# In[15]:


X


# ### One Hot Encoding Geography Feature
# * Here we have applied the one hot encoding because there are no autorelation between countries and multiple types names of countries.

# In[18]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


# In[19]:


print(X)


# ### Split the Dataset into Train and Test

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Feature Scaling
# * Feature Scaling is kind of a must for deep learning whenever you try to slove a problem using deep learning you should apply feature scaling.
# * We will feature scaling to every feature column
# * In Neural Networks you should apply the feature scaling on every feature but in machine learning it can vary.

# In[21]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# # Building The ANN Model

# In[26]:


ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu",name="layer1")) #First Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu",name="layer2")) #Secod Hidden Layer
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid",name="layer3")) #Output Layer 
# At the output layer the activation function for binary classification should be sigmoid
# For multi classificatio the activation function should be softmax.


# ## Training the model

# In[27]:


ann.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])  
#For binary classification the loss must be binay_crossentropy
#For Multi Classification the loss must be multi_crossentropy


# In[28]:


ann.fit(X_train,y_train,batch_size=32,epochs=100)


# In[29]:


print("Number of weights after calling the model:", len(ann.weights))


# In[30]:


ann.summary()


# # Making Predictions

# In[36]:


ann.predict(sc.transform([[1,0,0,1,600,1,45,10,60000,2,1,1,40000]]))
#Any input of the predict method should be 2d array
# 1.0 0.0 1.0 france from encoding dummy values
# Than the other features


# In[38]:


print(ann.predict(sc.transform([[1,0,0,1,600,1,45,10,60000,2,1,1,40000]]))>0.5)
#if the predicted result is greater than 0.5 than we consider that the customer will not stay and true otherwise false.


# * Therefore, our ANN model predicts that this customer stays in the bank!
# * Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# * Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

# In[42]:


y_pred=ann.predict(X_test)
y_pred=y_pred>0.5
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# # Confusion Matrix

# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

