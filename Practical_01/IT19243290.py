#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Coded By Wickramasinghe T.W.M.K.L
#IT19243290


# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[3]:


dataSet=pd.read_csv('Sales.csv')


# In[4]:


dataSet.head()


# In[5]:


dataSet.shape


# In[13]:


y=dataSet.iloc[:,0]


# In[14]:


y.head()


# In[18]:


x1=dataSet.iloc[:,1:2]


# In[19]:


x1.head()


# In[35]:


x2=dataSet.iloc[:,2:3]


# In[36]:


x2.head()


# In[23]:


x3=dataSet.iloc[:,3:4]


# In[24]:


x3.head()


# In[25]:


plt.scatter(x1,y)


# In[26]:


model = LinearRegression()
model.fit(x1, y)


# In[27]:


X = sm.add_constant(x1)
model = sm.OLS(y,X).fit()
model.summary()


# In[28]:


plt.scatter(x2,y)


# In[37]:


model1 = LinearRegression()
model1.fit(x2, y)


# In[38]:


X1 = sm.add_constant(x2)
model1 = sm.OLS(y,X1).fit()
model1.summary()


# In[31]:


plt.scatter(x3,y)


# In[32]:


model2 = LinearRegression()
model2.fit(x3, y)


# In[33]:


X2 = sm.add_constant(x3)
model1 = sm.OLS(y,X2).fit()
model1.summary()


# In[ ]:




