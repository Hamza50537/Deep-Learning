#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
input_array=np.array([5,6,7,8,9,10])
print(input_array)


# In[7]:


import time

alpha=np.random.rand(1000000)
beta=np.random.rand(1000000)

before=time.time()
output=np.dot(alpha,beta)
after=time.time()

print(output)

print("Vectorized Version: "+str(1000*(after-before))+"ms")

delta=0
before_for=time.time()
for i in range(1000000):
    delta+=alpha[i]*beta[i]
after_for=time.time()

print(delta)
print("For loop version: "+str(1000*(after_for-before_for))+"ms")


# So Vectorization version is 5 times better than for loop version. I hope that little example helps you in the understanding of vectorization concept>

# In[ ]:




