from Random_Forest import *

import random
import pandas as pd
import numpy as np

data = pd.read_csv('train_Loan_Info.csv')
data = data.drop(columns=['Loan_ID'])


data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Education'].fillna(data['Education'].mode()[0], inplace=True)
data['Property_Area'].fillna(data['Property_Area'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['ApplicantIncome'].fillna(data['ApplicantIncome'].median(), inplace=True)
data['CoapplicantIncome'].fillna(data['CoapplicantIncome'].median(), inplace=True)

data_w_dummies = pd.get_dummies(data, drop_first=True)
data_w_dummies.replace({False: 0, True: 1}, inplace=True)

features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Gender_Male', 'Married_Yes',
       'Dependents_1', 'Dependents_2', 'Dependents_3+',
       'Education_Not Graduate', 'Self_Employed_Yes',
       'Property_Area_Semiurban', 'Property_Area_Urban']

nb_train = int(np.floor(0.9 * len(data_w_dummies)))
data_w_dummies = data_w_dummies.sample(frac=1, random_state=217)
X_train = data_w_dummies[features][:nb_train]
y_train = data_w_dummies['Loan_Status_Y'][:nb_train].values
X_test = data_w_dummies[features][nb_train:]
y_test = data_w_dummies['Loan_Status_Y'][nb_train:].values

model = random_forest(X_train, y_train, n_estimators=20, max_features=3, max_depth=2, min_samples_split=2)

preds = predict_random_forest(model, X_test)
acc = sum(preds == y_test) / len(y_test)
print("Testing accuracy: {}".format(np.round(acc,3)))

