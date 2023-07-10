#!/usr/bin/env python
# coding: utf-8

# # Examples

# In[4]:


import pandas as pd
df = pd.read_csv('vgsales.csv')
df.shape


# In[6]:


df.describe()


# In[8]:


# Count -> Number of records in that column
df.values


# # Start of the project
# 
# ### Steps taken here
# #### 1) Import the dataset
# #### 2) Clean the data
# #### 3) Split data into training and testing
# #### 4) Create a model
# #### 5) Train the model
# #### 6) Make predictions
# #### 7) Evaluate and improve

# In[15]:


import pandas as pd
music_data = pd.read_csv('music.csv')
music_data


# In[17]:


# We split the data for training and testing
# Input dataset
X = music_data.drop(columns = ['genre'])
#Output dataset
y = music_data['genre']
y


# # Training and Predicting

# In[20]:


# Here we will be using Decision Tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
X = music_data.drop(columns = ['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X,y)
# It takes the input and output datasets as inputs
music_data


# #### Here we can see that in this dataset we have provided that men within age 20 - 25 like hiphop so if we ask our model to predict 
# #### the genre for a 22 year old male it should give the output as hiphop

# In[24]:


predictions = model.predict([[21, 1], [22,0]])
predictions
# It takes a 2d array as input


# ## Measuring its accuracy

# In[25]:


# We cannot measure it's accuracy out of just these 2 outputs
# We need to train the model with atleast 70 - 80 % of the data and test with the remaining data


# In[26]:


# So we use train test split
from sklearn.model_selection import train_test_split


# In[90]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns = ['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'music_recommender.joblib')
# predictions = model.predict(X_test)
# For comparison or accuracy measure we compare predicitions with y_test
# For accuracy we import metrics library of sklearn
# score = accuracy_score(y_test, predictions)
# First input is the expected values and second is our output
#Ctrl + enter to run it multiple times
# score


# In[88]:


# Storing the model as it is very impractical to train a model multiple times
# For this we use the joblib library 
from sklearn.externals import joblib
# Instead of this we directly import joblib to avoid error that was there idk why
joblib.dump(model, 'music_recommender.joblib')


# In[93]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns = ['genre'])
# y = music_data['genre']
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# Now instead of dumping we load the model
model = joblib.load('music_recommender.joblib')
predictions = model.predict([[21,1]])
predictions


# In[94]:


# Visualisation of Decision Tree
# For this we use the tree library directly
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
X = music_data.drop(columns = ['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X,y)
tree.export_graphviz(model, out_file = 'music-recommender.dot', feature_names = ['age', 'gender'], 
                     class_names = sorted(y.unique()), label = 'all', rounded = True, filled = True)


# In[ ]:


# We then use vscode extension to open this dot file and visualize it as a decision tree

