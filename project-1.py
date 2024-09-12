#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')


# Loading Dataset

# In[9]:


wine=pd.read_csv("D:\winequality-red.csv")
wine.sample(25)


# In[10]:


wine.info()


# Description

# In[11]:


wine.describe()


# Finding Null Values

# In[13]:


print(wine.isnull().sum())


# In[14]:


wine.groupby('quality').mean()


# Data Analysis

# Countplot:

# In[15]:


sns.countplot(wine['quality'])
plt.show()


# In[16]:


wine.plot(kind='box',subplots=True,layout=(4,4),sharex=False)


# In[17]:


wine['fixed acidity'].plot(kind='box')


# Histogram

# In[20]:


wine.hist(figsize=(10,10),bins=50)
plt.show()         


# Feature Selection

# In[21]:


wine.sample(5)


# In[22]:


wine['quality'].unique()


# In[23]:


# if wine quality is 7 or above then will consider as good quality wine
wine['goodquality']=[1 if x>=7 else 0 for x in wine ['quality']]
wine.sample(5)


# In[25]:


# Separate dependent and independent variable
x=wine.drop(['quality','goodquality'], axis=1)
y=wine['goodquality']


# In[26]:


# See total number of good vs bad wines samples
wine['goodquality'].value_counts()


# In[27]:


x
print(y)


# Feature Importance

# In[28]:


from sklearn.ensemble import ExtraTreesClassifier
classifiern=ExtraTreesClassifier()
classifiern.fit(x,y)
score=classifiern.feature_importances_
print(score)


# Splitting Dataset

# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=7)


# Result

# In[37]:


model_res = {
  "Model": [],
  "Score": []
}


# In[42]:


model_res=pd.DataFrame(model_res)


# LogisticRegression

# In[47]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)



# In[48]:


from sklearn.metrics import accuracy_score,confusion_matrix
#accuracy_score(y_test,y_pred)

model_res.loc[len(model_res)]=['LogisticRegression',accuracy_score(y_test,y_pred)]
model_res


# In[51]:


pip install -U scikit-learn


# Using KNN

# In[55]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[56]:


from sklearn.metrics import accuracy_score
model_res.loc[len(model_res)]=['kNeighborClassifier',accuracy_score(y_test,y_pred)]
model_res


# Using SVC

# In[58]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[59]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)]=['SVC',accuracy_score(y_test,y_pred)]


# Using Decision Tree

# In[60]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[61]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)]=['DecisionTreeClassifier',accuracy_score(y_test,y_pred)]


# Using GaussianNB

# In[62]:


from sklearn.naive_bayes import GaussianNB
model3=GaussianNB()
model3.fit(x_train,y_train)
y_pred=model3.predict(x_test)


# In[63]:


from sklearn.metrics import accuracy_score
print("Accuracy score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)]=['GaussianNB',accuracy_score(y_test,y_pred)]
model_res


# Using Random Forest

# In[64]:


from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(random_state=1)
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)


# In[71]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)]=['RandomForestClassifier', accuracy_score(y_test,y_pred)]
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)]=['RandomForestClassifier',accuracy_score(y_test,y_pred)]

model_res=model_res.sort_values(by='Score',ascending=False)
model_res


# In[ ]:




