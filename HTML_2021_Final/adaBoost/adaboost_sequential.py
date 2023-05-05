import numpy as np
import pandas as pd
from csvfiledata import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn import metrics

training = pd.read_csv('result_train.csv')

numerical = [
    'Age', 'Number of Dependents', 'Population', 'Latitude', 'Longitude', 'Satisfaction Score',
    'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 
    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']

for feature in numerical:
    median = np.nanmedian(training[feature])
    new_col = np.where(training[feature].isnull(), median, training[feature])
    training[feature] = new_col

X = pd.DataFrame([training[key] for key in training.keys() if key != 'Churn Category']).T
y = training["Churn Category"]


X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=3)

y_train = y_train.tolist()
y_validate = y_validate.tolist()
boost = [0]*6

for target in range((6)):
    y = [0 for _ in y_train]
    for i in range(len(y_train)):
        if y_train[i] == target:
            y[i] = 1
        else:
            y[i] = 0

    _y = [0 for _ in y_validate]
    for i in range(len(y_validate)):
        if y_validate[i] == target:
            _y[i] = 1
        else:
            _y[i] = 0

    boost[target] = AdaBoostClassifier(n_estimators = 19)
    boost[target].fit(X_train, y)
    
    test_y_predicted = boost[target].predict(X_validate)
    accuracy = accuracy_score(_y, test_y_predicted)
    print("Accuracy", target ,":",accuracy)

# predict and accuracy

"""==========train above===test below=========="""

header, content = get_file('../sample_submission.csv')
testing = pd.read_csv('result_test.csv')

for feature in numerical:
    median = np.nanmedian(testing[feature])
    new_col = np.where(testing[feature].isnull(),median, testing[feature])
    testing[feature] = new_col

X_test = pd.DataFrame([testing[key] for key in testing.keys() if key != 'Churn Category']).T




order = [5,4,2,3,1,0]
prediction = [0]*6
overall_prediction = [0]*len(X_validate)

for i in range(6):
    prediction[i] = boost[i].predict(X_validate)
for i in range(len(X_validate)):
    for target in order:
        if prediction[target][i] == 1:
            overall_prediction[i] = target
            break


print(metrics.classification_report(y_validate, overall_prediction))
print("Confusion matrix")
print(metrics.confusion_matrix(y_validate, overall_prediction))

prediction = [0]*6
overall_prediction = [0]*len(X_test)

for i in range(6):
    prediction[i] = boost[i].predict(X_test)
for i in range(len(X_test)):
    for target in order:
        if prediction[target][i] == 1:
            overall_prediction[i] = target
            break

for i in range(len(content)): content[i][1] = overall_prediction[i]


write_file('predict_adaboost.csv', header, content)
