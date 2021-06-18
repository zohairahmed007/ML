# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:18:07 2021

@author: Zohair
"""
# =============================================================================
# SMS Spam Collection data set is taken from the UCI Machine Learning Repository. This data set is a public set of SMS labeled messages that were collected for mobile phone spam research in 2012. It consists of 5572 messages of which 4825 are ham messages and 747 spam messages. In this data set, every line starts with the label of the message, followed by the text. 
# Dataset link: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# =============================================================================

 # Importing packages:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_columns',None)
pd.set_option('max_rows',None)

import sys
np.set_printoptions(threshold=sys.maxsize)

# Reading the dataset:

df =pd.read_csv("Dataset//spamdata.csv",nrows=1000)
df.head()


#Plot Labels
print("Class Labels:\n",df['label'].value_counts())
df['label'].value_counts().plot(kind='bar')


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

print("Stop Words:\n",set(stopwords.words('english')))
 
filter_sent=[]
stm= PorterStemmer()

for i in range(len(df)):
    s= df.iloc[i][1]
    s=s.lower()
    s=s.split()
    
    s=[stm.stem(w) for w in s if w not in set(stopwords.words('english'))]
    
    s=' '.join(s)
    
    filter_sent.append(s)
   
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()

X=cv.fit_transform(filter_sent).toarray()
print(X.shape)
print(cv.vocabulary_)

y=df['label']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=150)


from sklearn.svm import LinearSVC
linearsvc = LinearSVC()
linearsvc.fit(x_train,y_train)

y_pred = linearsvc.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score

print("Classification Report:\n",classification_report(y_test,y_pred))
print("Accuracy Score:\n",accuracy_score(y_test,y_pred))
#Get the confusion matrix
cm=confusion_matrix(y_test,y_pred,labels=["ham","spam"])
print("Confusion Matrix :\n",cm)
 
list1 = ["Actual ham", "Actual spam"]
list2 = ["Predicted ham", "Predicted spam"]
print(pd.DataFrame(cm, list1,list2))

# Testing New Data
# =============================================================================
# newtest=["free tickets for rawalpindi stadium"]
# 
# transform_newtest=cv.transform(newtest).toarray()
# 
# print(linearsvc.predict(transform_newtest))
# 
# =============================================================================


