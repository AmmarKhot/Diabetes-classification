#!/usr/bin/env python
# coding: utf-8

# # DIABETES CLASSIFICATION USING RANDOM FOREST ALGORITHEM

# ## IMPORTING LIBRARY

# In[1]:


import numpy as np
import pandas as pd


# ## IMPORTING DATASET

# In[2]:


dataset = pd.read_csv('diabetes.csv')


# In[3]:


dataset


# ## CHECKENING HOW MANY 0 ARE IN DATASET

# In[4]:


print((dataset['Glucose']==0).sum())
print((dataset['BloodPressure']==0).sum())
print((dataset['SkinThickness']==0).sum())
print((dataset['Insulin']==0).sum())
print((dataset['BMI']==0).sum())
print((dataset['DiabetesPedigreeFunction']==0).sum())
print((dataset['Age']==0).sum())


# ## FINDING THE MEAN VALUE OF THE COLUMN

# In[5]:


print(dataset['Glucose'].mean())
print(dataset['BloodPressure'].mean())
print(dataset['SkinThickness'].mean())
print(dataset['Insulin'].mean())
print(dataset['BMI'].mean())


# ## REPLACING 0 WITH THE MEAN VALUE OF THAT COLUMN

# In[6]:


dataset['Glucose'] = dataset['Glucose'].replace(0,121.1825)
dataset['BloodPressure'] = dataset['BloodPressure'].replace(0,69.1455)
dataset['SkinThickness'] = dataset['SkinThickness'].replace(0,20.935)
dataset['Insulin'] = dataset['Insulin'].replace(0,80.254)
dataset['BMI'] = dataset['BMI'].replace(0,32.192999999999984)


# In[7]:


dataset


# ## LOCATING VARIABLES AND LABLES

# In[8]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## DIVIDING DATASET INTO TRANING AND TASTING SETS

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =7)


# In[10]:


print(X_train)


# In[11]:


print(X_test)


# ## FEATURE SCALING

# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[13]:


print(X_train)


# In[14]:


print(X_test)


# ## IMPORTING THE ALGORITHEM

# In[15]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5,max_depth=10, criterion = 'entropy', random_state = 10)
classifier.fit(X_train, y_train)


# ## COMPARING PRIDICTED RESULT WITH ACTUAL RESULT

# In[16]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## CREATING CONFUSION MATRIX

# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score,mean_absolute_error
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ## ANALYSING THE RESULT

# In[18]:


#traning accuracy
Yp_train = classifier.predict(X_train)
(Yp_train == y_train).sum()/len(X_train)


# In[19]:


#tseting accuracy
Yp_test = classifier.predict(X_test)
(Yp_test == y_test).sum()/len(X_test)


# In[20]:


mean_absolute_error(y_test,y_pred)


# In[ ]:




