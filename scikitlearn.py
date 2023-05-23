#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### scikit-learn

"""
the package builds ML 
has features like; classification, clustering & regression

the package has algorithms like; k-means, support vector machine(SVM), K NEAREST NEIGHBORS, & DECISION TREES

PROBLEM-SOLUTION APPROACH
"""


# In[2]:


# clustering
# cross-validation
# ensemble methods
# feature extraction
# feature selection
# parameter tuning
# supervised learning algorithm like decision tree, SVM and regression


# In[5]:


"""
PROBLEM STATMENT:
You have been provided with a dataset that contains the costs of advertising on different
media channels and the corresponding sales of XYZ firm. Evaluate the dataset to:
1. Find the features or media channels used by the firm
2. Find the salas figures for each channel
3. Create a model to predict teh sales outcome
4. Split it into training and testing datasets for the model
5. Calculate the mean squared error (MSE)
"""


# In[4]:


import pandas as pd


# In[7]:


data = pd.read_csv("C:/Users/Ayieko/Desktop/python/simplilearn/advertising/Advertising.csv", index_col=0)
print(data)


# In[8]:


data.size ### view top 5 records


# In[9]:


data.shape ## view the shape of teh dataset


# In[10]:


data.columns ### view teh colums of the dataset


# In[11]:


x_feature = data[['Newspaper', 'Radio', 'TV']] ### create a feature object from teh colums


# In[12]:


x_feature.head() ### view feature object


# In[13]:


y_target = data[['Sales']] #### create a target object from sales column which is a response in the dataset


# In[14]:


y_target.head() ### view the target object


# In[16]:


### view the target object shape
y_target.shape


# In[27]:


### split test and training dataset
### by default, 75% of the training data and 25% testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target, random_state=1)
print(x_train.shape) ### views the train set
print(y_train.shape)  ### views the train set
print(x_test.shape) ### views the test set
print(y_test.shape) ### views the test set


# In[38]:


#### linear regression 
from sklearn.linear_model import LinearRegression


# In[37]:


linreg =LinearRegression()
linreg.fit(x_train,y_train)


# In[40]:


### print coefficients and intercept
print(linreg.intercept_)
print(linreg.coef_)


# In[43]:


### prediction
y_pred = linreg.predict(x_test)
y_pred


# In[44]:


## a better way of prediction
from sklearn import metrics
import numpy as np


# In[47]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:


###

