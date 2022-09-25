#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans 
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


data=pd.read_excel("D:\project github\project 2\data costumers.xlsx")


# In[3]:


data.head()


# # Univariate Analysis 

# In[4]:


data.columns


# In[5]:


data.describe()


# In[6]:


sns.distplot(data['Annual Income (k$)']);


# In[11]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(data[i])


# In[17]:


sns.kdeplot(data['Annual Income (k$)'], shade=True, hue=data['Gender'])


# In[19]:


columns =['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data[i], shade=True, hue=(data['Gender']))


# In[20]:


columns =['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=data, x='Gender', y=data[i])


# In[25]:


data['Gender'].value_counts(normalize=True)


# # Bivariats Analysis

# In[26]:


sns.scatterplot(data=data, x='Annual Income (k$)',y='Spending Score (1-100)' )


# In[27]:


sns.pairplot(data,hue='Gender')


# In[28]:


data.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[29]:


data.corr()


# In[30]:


sns.heatmap(data.corr(),annot=True,cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate

# In[31]:


clustering1 = KMeans(n_clusters=3)


# In[32]:


clustering1.fit(data[['Annual Income (k$)']])


# In[33]:


clustering1.labels_


# In[34]:


data['Income Cluster'] = clustering1.labels_
data.head()


# In[35]:


data['Income Cluster'].value_counts()


# In[36]:


clustering1.inertia_


# In[37]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[38]:


intertia_scores


# In[39]:


plt.plot(range(1,11),intertia_scores)


# In[40]:


data.columns


# In[41]:


data.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Bivariate Clustering

# In[42]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(data[['Annual Income (k$)','Spending Score (1-100)']])
data['Spending and Income Cluster'] =clustering2.labels_
data.head()


# In[43]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(data[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[44]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[45]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=data, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[46]:


pd.crosstab(data['Spending and Income Cluster'],data['Gender'],normalize='index')


# In[47]:


data.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[48]:


#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[49]:


scale = StandardScaler()


# In[50]:


data.head()


# In[51]:


dff = pd.get_dummies(data,drop_first=True)
dff.head()


# In[52]:


dff.columns


# In[53]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[54]:


dff = scale.fit_transform(dff)


# In[55]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[56]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[57]:


data


# In[58]:


data.to_excel('Clustering.xlsx')

