# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:16:17 2019

@author: Hong
"""

# ANN
# Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.getcwd()
os.chdir('C:\\Users\\Hong\\Documents\\P16-Deep-Learning-AZ\\P16-Artificial-Neural-Networks\\20190909')

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoiding dummy variable trap!

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Par 2 - Now let's make the ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
# Number of hidden layer nodes: (11(number of input) + 1(number of output))/2
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to the Training set
# classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)
classifier.fit(X_train, y_train, batch_size = 5, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0, 0] + cm[1, 1])/cm.sum()
accuracy

# Predicting a new data
RowNumber = [0]
CustomerId = [0]
Surname = ["name"]
CreditScore = [600]
Geography = ["France"]
Gender = ["Male"]
Age = [40]
Tenure = [3]
Balance = [60000]
NumOfProducts = [2]
HasCrCard = [1]
IsActiveMember = [1]
EstimatedSalary = [50000]

X_new_dict = {
        'RowNumber':RowNumber,
        'CustomerId':CustomerId,
        'Surname':Surname,
        'CreditScore':CreditScore,
        'Geography':Geography,
        'Gender':Gender,
        'Age':Age,
        'Tenure':Tenure,
        'Balance':Balance,
        'NumOfProducts':NumOfProducts,
        'HasCrCard':HasCrCard,
        'IsActiveMember':IsActiveMember,
        'EstimatedSalary':EstimatedSalary   
}
X_new_df = pd.DataFrame(X_new_dict)
X_new = X_new_df.iloc[:, 3:13].values

X_new[:, 1] = labelencoder_X_1.transform(X_new[:, 1])
X_new[:, 2] = labelencoder_X_2.transform(X_new[:, 2])
X_new = onehotencoder.transform(X_new).toarray()
X_new = X_new[:, 1:] # Avoiding dummy variable trap!
X_new = sc.transform(X_new)

y_new_pred = classifier.predict(X_new)
y_new_pred = (y_new_pred > 0.5)
y_new_pred

# Example
a = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
a = sc.transform(a)
a_pred = classifier.predict(a)
a_pred = (a_pred > 0.5)
a_pred

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
accuracies.mean()
accuracies.std()
accuracies.std()/accuracies.mean() * 100

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 50], 
              'epochs': [100], 
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
a = grid_search.cv_results_



# =========================================================================



# Fitting the classifier to the Training set
# Create your classifier here (Ex: Logistic Regression, SVM ...)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, y_train)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()
classifier.fit(X_train, y_train)


# Predicting the Test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, classifier.predict(X_train))
cm_test = confusion_matrix(y_test, y_pred)
accuracy_train = (cm_train[0, 0] + cm_train[1, 1])/cm_train.sum()
accuracy_test = (cm_test[0, 0] + cm_test[1, 1])/cm_test.sum()


