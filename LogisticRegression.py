# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset

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


# Fitting the classifier to the Training set
# Create your classifier here (Ex: Logistic Regression, SVM ...)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
def Classification(clf, X, y):
    X_set, y_set = X, y
    y_hat = clf.predict(X_set)
    y_hat = np.reshape(y_hat, -1)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    cm = confusion_matrix(y_set, y_hat)
    accuracy = (cm[0, 0] + cm[1, 1])/cm.sum()
    TPR = cm[0, 0]/cm[:, 0].sum() # Sensitivitive, Recall
    TNR = cm[1, 1]/cm[:, 1].sum() # Specificitive
    PPV = cm[0, 0]/cm[0, :].sum() # Positive Predictive Value, Precision
    NPV = cm[1, 1]/cm[1, :].sum() # Negative Predictive Value,  
    F1_score = 2/(1/PPV + 1/TPR)
    summary = {'Accuracy': accuracy, 
               'Positive_Predictive_Value': PPV, 
               'Negative_Predictive_Value': NPV,            
               'Sensitivitive': TPR, 
               'Specificitive': TNR,            
               'F1_score': F1_score, 
               'CM': cm}
    return summary

Classification(clf = classifier, X = X_train, y = y_train)
Classification(clf = classifier, X = X_test, y = y_test)


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
parameter = [{'penalty': ['l1'], 'C': np.arange(0.0001, 0.1, 0.0001)}, 
             {'penalty': ['l2'], 'C': np.arange(0.0001, 0.1, 0.0001)}]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameter,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
results = grid_search.cv_results_
best_parameters


# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C = best_parameters['C'], 
                                penalty = best_parameters['penalty'], 
                                random_state = 0)
accuracies = cross_val_score(estimator = classifier,
                             X = X_train, y = y_train, 
                             cv = 10)
plt.plot(accuracies, '--o')
plt.axhline(accuracies.mean(), color = 'black')
plt.show()



# Plot L1 coefficients
from sklearn.linear_model import LogisticRegression
p = len(X_train[0])
c = np.arange(0.0001, 0.1, 0.001)
Cost = np.zeros(shape = (len(c), p))
coef = np.zeros(shape = (len(c), p))
for ii in range(len(c)):
    classifier = LogisticRegression(penalty = 'l1', 
                                    C = c[ii], 
                                    random_state = 0)
    classifier.fit(X_train, y_train)
    coef[ii, :] = classifier.coef_
    Cost[ii, :] = np.full((1, p), c[ii])

for i in range(p):
    plt.plot(np.log(Cost[:, i]), coef[:, i], '-o', label = i)
plt.axhline(0, color = 'black')
plt.axvline(np.log(best_parameters['C']), ls = '--', color = 'black')
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.legend()
plt.show()



# Fitting the classifier to the Training set with best_parameters
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C = best_parameters['C'], 
                                penalty = best_parameters['penalty'], 
                                random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
Classification(clf = classifier, X = X_train, y = y_train)
Classification(clf = classifier, X = X_test, y = y_test)


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
        'RowNumber': RowNumber,
        'CustomerId': CustomerId,
        'Surname': Surname,
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary   
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

