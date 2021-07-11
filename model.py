#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv("C:\\Users\\Pc\\Datasets\\iris.csv")


# In[3]:


data.head()


# In[4]:


data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].hist()


# In[5]:


X = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]


# In[6]:


y = data[["Species"]]


# In[7]:


X_train, X_test, y_train,y_test = train_test_split(X, y , test_size=0.2, random_state=4)


# In[8]:


lr = LogisticRegression()


# In[9]:


KNN = KNeighborsClassifier(n_neighbors=10)


# In[10]:


DTC = DecisionTreeClassifier(random_state=0)


# In[11]:


y_train.shape


# In[12]:


X_train.shape


# In[13]:


lr.fit(X_train, y_train)


# In[14]:


KNN.fit(X_train, y_train)


# In[15]:


DTC.fit(X_train, y_train)


# In[16]:


yhatlr = lr.predict(X_test)
yhatKNN = KNN.predict(X_test)
yhatDTC = DTC.predict(X_test)


# In[17]:


print(accuracy_score(yhatlr, y_test))
confusion_matrix(yhatlr, y_test)


# In[18]:


print(accuracy_score(yhatKNN, y_test))
confusion_matrix(yhatKNN, y_test)


# In[19]:


print(accuracy_score(yhatDTC, y_test))
confusion_matrix(yhatDTC, y_test)


# In[20]:


import pickle


# In[21]:


pickle.dump(KNN, open('model.pkl','wb'))


# In[24]:


model = pickle.load(open('model.pkl','rb'))


# In[25]:


model.predict([[2.3, 4.5, 6.7, 8.7]])


# In[ ]:




