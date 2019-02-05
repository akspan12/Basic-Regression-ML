# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 18:38:47 2018

@author: AKSPAN12
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#spliting dataset in training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=1/3,random_state = 0) 

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting simple linear regression on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the test set result
y_pred = regressor.predict(X_test)

#visualising the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()



