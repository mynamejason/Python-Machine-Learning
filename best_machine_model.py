# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:45:05 2020

@author: Jason
"""

import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep = ";")                #Delimiter/Separator
print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

forecast = "G3"

x = np.array(data.drop([forecast], 1))
y = np.array(data[forecast])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

high_score = 0

for i in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)
    
    if accuracy > high_score:
        high_score = accuracy
        with open("student_model.pickle", "wb") as f:                   #"wb": write binary
            pickle.dump(linear, f)
    
pickle_read = open("student_model.pickle", "rb")
linear = pickle.load(pickle_read)

#print(linear)
#print(accuracy)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])