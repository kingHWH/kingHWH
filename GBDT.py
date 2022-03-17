'''
开发者：HAIGEMAY
开发时间：2021/12/23 14:43

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# from sklearn.grid_search import GridSearchCV


train = pd.read_csv('train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts()
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X,y)

y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1]
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

