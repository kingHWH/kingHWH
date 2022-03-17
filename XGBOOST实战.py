'''
开发者：HAIGEMAY
开发时间：2021/12/25 9:43

'''
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.datasets.samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#第一种
sklearn_model_new = xgb.XGBClassifier(n_estimators=10,max_depth=5,learning_rate= 0.5, verbosity=1, objective='binary:logistic',random_state=1)
sklearn_model_new.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",
        eval_set=[(X_test, y_test)])

#第二种
# dtrain = xgb.DMatrix(X_train,y_train)
# dtest = xgb.DMatrix(X_test,y_test)
#
# param = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
# raw_model = xgb.train(param, dtrain, num_boost_round=20)
# from sklearn.metrics import accuracy_score
# pred_train_raw = raw_model.predict(dtrain)
# for i in range(len(pred_train_raw)):
#     if pred_train_raw[i] > 0.5:
#          pred_train_raw[i]=1
#     else:
#         pred_train_raw[i]=0
# print (accuracy_score(dtrain.get_label(), pred_train_raw))
#
# pred_test_raw = raw_model.predict(dtest)
# for i in range(len(pred_test_raw)):
#     if pred_test_raw[i] > 0.5:
#          pred_test_raw[i]=1
#     else:
#         pred_test_raw[i]=0
# print (accuracy_score(dtest.get_label(), pred_test_raw))