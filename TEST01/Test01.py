#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[2]:


dataSet=pd.read_csv('CarPrice.csv')


# In[3]:


dataSet.head()


# In[4]:


dataSet.shape


# In[5]:


y=dataSet.iloc[:,6]


# In[6]:


y.head()


# In[7]:


dataSet['fuelType']=dataSet['fuelType'].factorize()[0]


# In[10]:


dataSet.head()


# In[9]:


dataSet['cylinders']=dataSet['cylinders'].factorize()[0]


# In[20]:


x1=dataSet.iloc[:,0:1]


# In[21]:


x1.head()


# In[13]:


x2=dataSet.iloc[:,1:2]


# In[14]:


x2.head()


# In[15]:


x3=dataSet.iloc[:,2:3]


# In[17]:


x3.head()


# In[19]:


x4=dataSet.iloc[:,3:4]


# In[22]:


x4.head()


# In[23]:


x5=dataSet.iloc[:,4:5]


# In[24]:


x5.head()


# In[25]:


x6=dataSet.iloc[:,5:6]


# In[26]:


x6.head()


# In[27]:


plt.scatter(x1,y)


# In[28]:


model = LinearRegression()
model.fit(x1, y)


# In[29]:


X = sm.add_constant(x1)
model = sm.OLS(y,X).fit()
model.summary()


# In[30]:


plt.scatter(x2,y)


# In[32]:


model1 = LinearRegression()
model1.fit(x2, y)


# In[33]:


X1 = sm.add_constant(x2)
model1 = sm.OLS(y,X1).fit()
model1.summary()


# In[34]:


plt.scatter(x3,y)


# In[35]:


model2 = LinearRegression()
model2.fit(x3, y)


# In[36]:


X2 = sm.add_constant(x3)
model1 = sm.OLS(y,X2).fit()
model1.summary()


# In[37]:


plt.scatter(x4,y)


# In[38]:


model3 = LinearRegression()
model3.fit(x4, y)


# In[39]:


X3 = sm.add_constant(x4)
model1 = sm.OLS(y,X3).fit()
model1.summary()


# In[40]:


plt.scatter(x5,y)


# In[41]:


model4 = LinearRegression()
model4.fit(x5, y)


# In[42]:


X4 = sm.add_constant(x5)
model1 = sm.OLS(y,X4).fit()
model1.summary()


# In[43]:


plt.scatter(x6,y)


# In[44]:


model5 = LinearRegression()
model5.fit(x6, y)


# In[45]:


X5 = sm.add_constant(x6)
model1 = sm.OLS(y,X5).fit()
model1.summary()


# In[ ]:




