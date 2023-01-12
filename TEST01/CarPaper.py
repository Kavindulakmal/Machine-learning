#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[41]:


#1
raw_data=pd.read_csv("movie_dataset.csv") 


# In[42]:


#2
raw_data.shape


# In[43]:


raw_data.head()


# In[44]:


#3
col_num=0  
TotalObjects = raw_data.shape[0]  
print ("Column\t\t\t\t\t Null Values%")  
for x in raw_data:
    nullCount = raw_data[x].isnull().sum();
    nullPercent = nullCount*100 / (TotalObjects)
    if nullCount > 0 and nullPercent > 20 :
        col_num=col_num+1
        #raw_data.drop(x, axis=1,inplace=True)
        print(str(x)+"\t\t\t\t\t "+str(nullPercent))  
print ("A total of "+str(col_num)+" found !") 


# In[45]:


raw_data= raw_data.drop(['The Revenant', '13 Hours', 'Allied','Jigsaw','Achorman','Grinch','Ghostbusters','Wolverine','Mad Max','John Wick'], axis =1) 


# In[46]:


raw_data.head()


# In[47]:


#4
raw_data['Zootopia'].fillna('10',inplace = True) 
raw_data['Fast and Furious'].fillna('10',inplace = True)
raw_data['La La Land'].fillna('10',inplace = True)
raw_data['The Good Dunosaur'].fillna('10',inplace = True)
raw_data['Ninja Turtles'].fillna('10',inplace = True)
raw_data['The Good Dunosaur Bad Moms'].fillna('10',inplace = True)
raw_data['2 Guns'].fillna('10',inplace = True)
raw_data['Inside Out'].fillna('10',inplace = True)
raw_data['Valerian'].fillna('10',inplace = True)
raw_data['Spiderman 3'].fillna('10',inplace = True)


# In[48]:


raw_data.head()


# In[49]:


import seaborn as sns 
sns.set() 
from sklearn.cluster import KMeans 


# In[50]:


#5
kmeans = KMeans(3) 
kmeans.fit(raw_data) 


# In[ ]:




