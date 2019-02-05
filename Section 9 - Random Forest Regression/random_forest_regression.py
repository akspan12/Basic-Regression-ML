# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:51:45 2018

@author: AKSPAN12
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values



#Fitting random forest regression model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

#predict using random forest regression
y_pred = regressor.predict(6.5)


#visualizing the random forest regression for higher resolution and smooth curve
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('truth or bluff(random forest regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


