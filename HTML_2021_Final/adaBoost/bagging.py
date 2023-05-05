import numpy as np
import pandas as pd
from csvfiledata import *
from sklearn.ensemble import BaggingClassifier
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
    new_col = np.where(training[feature].isnull(),
                    median, training[feature])
    training[feature] = new_col

X = pd.DataFrame([training[key] for key in training.keys() if key != 'Churn Category']).T
y = training["Churn Category"]

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)


bag = BaggingClassifier(n_estimators = 19)
bag_fit = bag.fit(X_train, y_train)

test_y_predicted = bag.predict(X_validate)
# accuracy = accuracy_score(y_validate, test_y_predicted)

# print('accuracy', accuracy)
# macroAvg = metrics.classification_report(y_validate, test_y_predicted).split('\n')[10].split()
# print(macroAvg[4], flush = True)
print(metrics.classification_report(y_validate, test_y_predicted))
# print("Confusion matrix")
# print(metrics.confusion_matrix(y_validate, test_y_predicted))



# training_multi = pd.read_csv('result_train_multi.csv')

# numerical = [
#     'Age', 'Number of Dependents', 'Population', 'Latitude', 'Longitude', 'Satisfaction Score',
#     'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges',
#     'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 
#     'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']

# for feature in numerical:
#     median = np.nanmedian(training_multi[feature])
#     new_col = np.where(training_multi[feature].isnull(),
#                     median, training_multi[feature])
#     training_multi[feature] = new_col

# X = pd.DataFrame([training_multi[key] for key in training_multi.keys() if key != 'Churn Category']).T
# y = training_multi["Churn Category"]
# X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=1)

# bag2 = BaggingClassifier(n_estimators = 100)
# bag_fit = bag2.fit(X_train, y_train)

# test_y_predicted = bag2.predict(X_validate)
# accuracy = accuracy_score(y_validate, test_y_predicted)

# print('accuracy', accuracy)
# # predict and accuracy
# # test_y_predicted = bag.predict(X_validate)
# # accuracy = accuracy_score(y_validate, test_y_predicted)

# """==========train above===test below=========="""

# header, content = get_file('../status.csv')

# testing = pd.read_csv('result_train_bin.csv')

# for feature in numerical:
#     median = np.nanmedian(testing[feature])
#     new_col = np.where(testing[feature].isnull(),
#                     median, testing[feature])
#     testing[feature] = new_col

# X_test = pd.DataFrame([testing[key] for key in testing.keys() if key != 'Churn Category']).T

# prediction =  bag.predict(X_test)

# not0 = [index for index in range(len(prediction)) if prediction[index] != 0]

# X_test_multi = pd.DataFrame([[testing[key][i] for i in range(len(testing[key])) if i in not0] for key in testing.keys() if key != 'Churn Category']).T

# prediction_multi =  bag2.predict(X_test_multi)

# for i in range(len(content)): content[i][1] = prediction[i] if prediction[i] == 0 else prediction_multi[not0.index(i)]


# training = pd.read_csv('result_train.csv')

# numerical = [
#     'Age', 'Number of Dependents', 'Population', 'Latitude', 'Longitude', 'Satisfaction Score',
#     'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges',
#     'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 
#     'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']

# for feature in numerical:
#     median = np.nanmedian(training[feature])
#     new_col = np.where(training[feature].isnull(),
#                     median, training[feature])
#     training[feature] = new_col

# X = pd.DataFrame([training[key] for key in training.keys() if key != 'Churn Category']).T
# y = training["Churn Category"]

# accuracy = accuracy_score(y, [row[1] for row in content])

# print(accuracy)
# write_file('guess_bagging.csv', header, content)
