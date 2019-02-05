# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:13:59 2018

@author: AKSPAN12
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)



#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y) 

#predict using regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualizing the SVR 
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('truth or bluff(SVR)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


#visualizing the svr for higher resolution and smooth curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('truth or bluff(regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()