#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


data = pd.read_csv('iris.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.dtypes


# In[8]:


data['species'].value_counts()


# In[9]:


data.corr()


# In[10]:


sns.pairplot(data, hue='species')
plt.show()


# In[11]:


sns.violinplot( x = 'sepal_length',y='species', data=data)


# In[12]:


sns.violinplot(x = 'sepal_width', y = 'species', data = data)


# In[13]:


sns.violinplot(x = 'petal_length', y = 'species', data = data)


# In[14]:


sns.violinplot(x = 'petal_width', y = 'species', data = data)


# In[15]:


X = data.drop(['species'], axis=1)
y = data['species']
print(X.shape)
print(y.shape)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[17]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()


# In[18]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[19]:


print(confusion_matrix(y_test,y_pred))


# In[20]:


print(classification_report(y_test, y_pred))

