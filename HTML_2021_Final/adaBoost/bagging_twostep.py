import numpy as np
import pandas as pd
from csvfiledata import *
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import sample
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

selectedIndex = sample(range(4226),3380)
notSelectedIndex = [i for i in range(4226) if i not in selectedIndex]

X_train = pd.DataFrame([[training[key][i] for i in selectedIndex] for key in training.keys() if key != 'Churn Category']).T
X_validate = pd.DataFrame([[training[key][i] for i in notSelectedIndex] for key in training.keys() if key != 'Churn Category']).T

y_train = [training["Churn Category"][i] for i in selectedIndex]
y_validate = [training["Churn Category"][i] for i in notSelectedIndex]


# y_train = y_train.tolist()
# y_validate = y_validate.tolist()

y_bin = [0 if item == 0 else 1 for item in y_train]
y_multi=  [item for item in y_train if item != 0]
X_multi = pd.DataFrame([[training[key][j] for i,j in enumerate(selectedIndex) if y_train[i] != 0] for key in training.keys() if key != 'Churn Category']).T

bag = BaggingClassifier(n_estimators = 19)
bag.fit(X_train, y_bin)
bag2 = BaggingClassifier(n_estimators = 19)
bag2.fit(X_multi, y_multi)

"""==========train above===test below=========="""

# header, content = get_file('../sample_submission.csv')
# testing = pd.read_csv('result_test.csv')

# for feature in numerical:
#     median = np.nanmedian(testing[feature])
#     new_col = np.where(testing[feature].isnull(),median, testing[feature])
#     testing[feature] = new_col

# X_test = pd.DataFrame([testing[key] for key in testing.keys() if key != 'Churn Category']).T

bin_prediction = bag.predict(X_validate)
not0 = [i for i in range(len(bin_prediction)) if bin_prediction[i] == 1]

X_multi = pd.DataFrame([[training[key][i] for i in not0] for key in training.keys() if key != 'Churn Category']).T

multi_prediction = bag2.predict(X_multi)

prediction = [0 if bin_prediction[i] == 0 else multi_prediction[not0.index(i)] for i in range(len(bin_prediction))]

# for i in prediction:
#     print(i)
print(metrics.classification_report(y_validate, prediction))
print("Confusion matrix")
print(metrics.confusion_matrix(y_validate, prediction))
