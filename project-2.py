#!/usr/bin/env python
# coding: utf-8

# Importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Data collection & analysis

# In[2]:


#loading the data from csv file to a pandas dataframe

insurance_dataset=pd.read_csv("D:\medical_cost_insurance.csv")


# In[3]:


#First 5 rows of the dataframe

insurance_dataset.head()


# In[4]:


#Number of rows and columns

insurance_dataset.shape


# In[5]:


#Getting some information about the dataset

insurance_dataset.info()


# In[6]:


#Checking for missing values

insurance_dataset.isnull().sum()


# Data Analysis

# In[7]:


#Statistical measures of the dataset

insurance_dataset.describe()


# In[8]:


#Distribution of age value

sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[9]:


#Gender column

plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[10]:


insurance_dataset['sex'].value_counts()


# In[11]:


#Bmi distribution

plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'])
plt.title('Age Distribution')
plt.show()


# In[12]:


# Children column

plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('children')
plt.show()


# In[13]:


insurance_dataset['children'].value_counts()


# In[14]:


#Smoker column

plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()


# In[15]:


insurance_dataset['smoker'].value_counts()


# In[16]:


#Region column

plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()


# In[17]:


insurance_dataset['region'].value_counts()


# In[18]:


#Distribution of charges value

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# Data Pre-processing

# Encoding the categorical features

# In[19]:


#Encoding 'sex' column

insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)


# In[20]:


#Encoding 'smoker'column

insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

#Encoding 'region' column

insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# Splitting the Features and Target

# In[21]:


X=insurance_dataset.drop(columns='charges',axis=1)

Y=insurance_dataset['charges']


# In[22]:


print(X)


# In[23]:


print(Y)


# In[25]:


#Splitting the data into training data and testing data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=2)







# In[27]:


print(X.shape,X_train.shape,X_test.shape)


# Model Training

# Linear Regression

# In[28]:


#Loading the Linear Regression model

regressor=LinearRegression()


# In[29]:


regressor.fit(X_train,Y_train)


# Model Evaluation

# In[30]:


#Prediction on training data

training_data_prediction=regressor.predict(X_train)


# In[31]:


#R squared value

r2_train=metrics.r2_score(Y_train, training_data_prediction)
print('R squared value:',r2_train)


# In[32]:


#prediction ontest data

test_data_prediction=regressor.predict(X_test)


# In[33]:


#R squared value

r2_test=metrics.r2_score(Y_test, test_data_prediction)
print('R squared value:',r2_test)


# Building a Predictive System

# In[35]:


input_data=(31,1,25.74,0,1,0)

#Changing input_data to a numpy array

input_data_as_numpy_array=np.asarray(input_data)

#Reshape tha array

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=regressor.predict(input_data_reshaped)

print(prediction)

print('The insurance cost is USD', prediction[0])


# In[ ]:




