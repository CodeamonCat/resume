import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from csvfiledata import *

from sklearn.model_selection import train_test_split

# 匯入訓練數據

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

X = pd.DataFrame([training[key]
                  for key in training.keys() if key != 'Churn Category']).T
y = training["Churn Category"]

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=12)

# 分類器使用 xgboost
# clf1 = xgb.XGBClassifier(eval_metric='mlogloss',
#                          use_label_encoder=False, nthread=15)
clf1 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.6666666666666666, enable_categorical=False, eval_metric='mlogloss', gamma=0, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.2, max_delta_step=0, max_depth=7,
                         min_child_weight=5, monotone_constraints='()', n_estimators=80, n_jobs=12, num_parallel_tree=1, objective='multi:softprob', predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=0.9, tree_method='exact', use_label_encoder=False, validate_parameters=1)

# 設定搜尋的xgboost參數搜尋範圍，值搜尋XGBoost的主要6個參數
param_dist = {
    'n_estimators': range(80, 200, 4),
    'max_depth': range(2, 15, 1),
    'learning_rate': np.linspace(0.01, 2, 20),
    'subsample': np.linspace(0.7, 0.9, 20),
    'colsample_bytree': np.linspace(0.5, 0.98, 10),
    'min_child_weight': range(1, 9, 1)
}

# RandomizedSearchCV參數說明，clf1設定訓練的學習器
# param_dist字典型別，放入參數搜尋範圍
# scoring = 'neg_log_loss'，精度評價方式設定爲「neg_log_loss「
# n_iter=300，訓練300次，數值越大，獲得的參數精度越大，但是搜尋時間越長
# n_jobs = -1，使用所有的CPU進行訓練，預設爲1，使用1個CPU
# grid = RandomizedSearchCV(clf1, param_dist, cv=5,
#                           scoring='f1_macro', n_iter=300, n_jobs=-1, verbose=10)
clf1.fit(X_train, y_train)

# 在訓練集上訓練
# grid.fit(X, y)
# 返回最優的訓練器
# best_estimator = grid.best_estimator_
best_estimator = clf1
# print(best_estimator,123)
# 輸出最優訓練器的精度
# print(grid.best_score_)


# header, content = get_file('../sample_submission.csv')
# testing = pd.read_csv('result_test.csv')

# for feature in numerical:
#     median = np.nanmedian(testing[feature])
#     new_col = np.where(testing[feature].isnull(), median, testing[feature])
#     testing[feature] = new_col

# X_test = pd.DataFrame([testing[key]
#                       for key in testing.keys() if key != 'Churn Category']).T

prediction = best_estimator.predict(X_validate)

# for i in prediction:
#     print(i)

print(metrics.classification_report(y_validate, prediction))
print("Confusion matrix")
print(metrics.confusion_matrix(y_validate, prediction))
# for i in range(len(content)):
#     content[i][1] = prediction[i]

# write_file('predict_xgboost.csv', header, content)
