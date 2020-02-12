#!/usr/bin/env python
# coding: utf-8

# In[227]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[246]:


data=pd.read_csv("D:DS_TriS/german_credit.csv")
data.head()


# In[247]:


data.columns


# In[248]:


sns.boxplot(data['Age (years)'])


# In[249]:


sns.boxplot(data['Credit Amount'])


# In[250]:


data.shape


# In[251]:


data.info()


# In[252]:


data['Age (years)']=np.log(data['Age (years)'])
data['Age (years)'].hist()


# In[253]:


data['Credit Amount']=np.log(data['Credit Amount'])
data['Credit Amount'].hist()


# In[254]:


sns.boxplot(data['Age (years)'])


# In[255]:


sns.boxplot(data['Credit Amount'])


# In[256]:


corr=data.corr()
corr.nlargest(20,'Creditability')['Creditability']


# In[257]:


x=data.drop(['Creditability'],1)
y=data['Creditability']
x=np.array(x)
y=np.array(y)


# In[258]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[259]:


x=sc.fit_transform(x)
x


# In[260]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=6)


# In[261]:


gnb = GaussianNB()
gnb.fit(x_train,y_train)
pred=gnb.predict(x_test)


# In[262]:


pred


# In[263]:


gnb.score(x_test,y_test)


# In[ ]:




