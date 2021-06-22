# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:25:01 2021

@author: Zohair
"""

# Colab Upload File ==========================================================
# from google.colab import files
# 
# uploaded = files.upload()
# 
# for fn in uploaded.keys():
#   print('User uploaded file "{name}" with length {length} bytes'.format(
#       name=fn, length=len(uploaded[fn])))
# =============================================================================

import pandas as pd
import io
#data=pd.read_csv(io.StringIO(uploaded['data.csv'].decode('utf-8')))
data=pd.read_csv("data.csv")
data.head()

import seaborn as sns
ax = sns.countplot(data['diagnosis'], label= 'Count')
B,M = data['diagnosis'].value_counts()
print('Benign', B)
print('Malignanat', M)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing data
del data['Unnamed: 32']

X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train

X_test


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#adding the input and first hidden layer
classifier = Sequential()
classifier.add(Dense(16, kernel_initializer='he_uniform', activation='relu',input_dim=30))

#classifier.add(Dropout(p=1.0))

#adding the second hidden layer
classifier.add(Dense(16, kernel_initializer='he_uniform', activation='relu'))

#adding the output layer
classifier.add(Dense(1, kernel_initializer='he_uniform', activation='sigmoid'))

classifier.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=100, epochs=150)

X_test

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True)
#plt.savefig('h.png')


