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

# pd.DataFrame(training).to_csv('median_fix_training_set.csv')
print('====')
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=3)

for i in range(1, 100):
    boost = AdaBoostClassifier(n_estimators = 19)
    boost.fit(X_train, y_train)

    test_y_predicted = boost.predict(X_validate)
    accuracy = accuracy_score(y_validate, test_y_predicted)
    macroAvg = metrics.classification_report(y_validate, test_y_predicted).split('\n')[10].split()
    print(str(i) + '\t'+macroAvg[4], flush = True)

    # print("Accuracy:",accuracy)
    
# print(metrics.classification_report(y_validate, test_y_predicted))
# print("Confusion matrix")
# print(metrics.confusion_matrix(y_validate, test_y_predicted))

"""==========train above===test below=========="""

header, content = get_file('../sample_submission.csv')
testing = pd.read_csv('result_test.csv')

for feature in numerical:
    median = np.nanmedian(testing[feature])
    new_col = np.where(testing[feature].isnull(),median, testing[feature])
    testing[feature] = new_col

X_test = pd.DataFrame([testing[key] for key in testing.keys() if key != 'Churn Category']).T

# pd.DataFrame(testing).to_csv('median_fix_testing_set.csv')

prediction = boost.predict(X_test)

for i in range(len(content)): content[i][1] = prediction[i]

# write_file('predict_adaboost.csv', header, content)
