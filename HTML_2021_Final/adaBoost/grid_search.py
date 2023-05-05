import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from csvfiledata import *


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


# 分類器使用 xgboost
clf1 = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)

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
grid = GridSearchCV(clf1, param_dist, cv=5, 
                          scoring='f1_macro',refit = True, n_jobs=-1, verbose=10)

# 在訓練集上訓練
grid.fit(X, y)
# 返回最優的訓練器
best_estimator = grid.best_estimator_
print(best_estimator)
# 輸出最優訓練器的精度
print(grid.best_score_)


header, content = get_file('../sample_submission.csv')
testing = pd.read_csv('result_test.csv')

for feature in numerical:
    median = np.nanmedian(testing[feature])
    new_col = np.where(testing[feature].isnull(),median, testing[feature])
    testing[feature] = new_col

X_test = pd.DataFrame([testing[key] for key in testing.keys() if key != 'Churn Category']).T

prediction = best_estimator.predict(X_test)

for i in range(len(content)): content[i][1] = prediction[i]

write_file('predict_xgboost.csv', header, content)
