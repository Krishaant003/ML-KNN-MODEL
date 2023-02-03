# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 19:26:40 2023

@author: krishaant.S.H
"""

#KNN - K nearest neighbor
#Predict  whether a person will have diabetics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

#showing datsets
dataset = pd.read_csv("diabetes.csv")
print (len(dataset))
print(dataset.head())

#replace zeros
zero_not_accepted =['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = int (dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN,mean)
    
#relationship between the datasj vkl
p=sns.pairplot(dataset, hue = 'Outcome')   


#split dataset
x=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

"""
Find K

import math
math.sqrt(len(y_test))

"""

#Define the model : Init K-NN
classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')

#fit model
classifier.fit(x_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='euclidean',
                     metric_params=None, n_jobs=1 , n_neighbors=11 , p = 2 , weights='uniform')

#predicting the test results
y_pred = classifier.predict(x_test)
y_pred


#Evaluating Model
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f1_score(y_test, y_pred))

print(accuracy_score(y_test,y_pred))


#visualizing the test results


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(x_train,y_train)
    
    train_scores.append(knn.score(x_train,y_train))
    test_scores.append(knn.score(x_test,y_test))

max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))    
    
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
    
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')