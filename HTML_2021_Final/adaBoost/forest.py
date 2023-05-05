import numpy as np
import pandas as pd
from csvfiledata import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

training = pd.read_csv('result_train_fair.csv')

numerical = [
    'Age', 'Number of Dependents', 'Population', 'Latitude', 'Longitude', 'Satisfaction Score',
    'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 
    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']

for feature in numerical:
    median = np.nanmedian(training[feature])
    new_col = np.where(training[feature].isnull(),
                    median, training[feature])
    training[feature] = new_col

X = pd.DataFrame([training[key] for key in training.keys() if key != 'Churn Category']).T
y = training["Churn Category"]
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=17)

for i in range(1,100):
    forest = RandomForestClassifier(n_estimators = i)
    forest.fit(X_train, y_train)



    test_y_predicted = forest.predict(X_validate)
    macroAvg = metrics.classification_report(y_validate, test_y_predicted).split('\n')[10].split()
    print(str(i) + '\t'+macroAvg[4], flush = True)
    # print(i, , flush = True)

