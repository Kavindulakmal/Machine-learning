#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import import the pandas, numpy and matplotlib.pyplot libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:





# In[3]:


import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[4]:


data=pd.read_csv("salary_data.csv")


# In[5]:


data.head


# In[6]:


data.shape


# In[7]:


# dislay only first 5 columns
data.head(5)


# In[11]:


#description of the salary
data["Salary"].describe()


# In[12]:


X=data.iloc[:,:-1].values


# In[13]:


X


# In[14]:


Y=data.iloc[:,1].values


# In[15]:


Y


# In[16]:


plt.scatter(X,Y)


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(X,Y)


# In[19]:


print(model.intercept_)


# In[20]:


print(model.coef_)


# In[21]:


model.predict(np.array([[5]]))


# In[22]:


x1 = sm.add_constant(X)


# In[23]:


model = sm.OLS(Y,x1).fit()
model.summary()


# In[ ]:




