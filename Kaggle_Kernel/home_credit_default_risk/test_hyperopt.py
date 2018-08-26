# from hpsklearn import HyperoptEstimator, svc
# from sklearn import svm
#
# # Load Data
# # ...
#
# if use_hpsklearn:
#     estim = HyperoptEstimator(classifier=svc('mySVC'))
# else:
#     estim = svm.SVC()
#
# estim.fit(X_train, y_train)
#
# print(estim.score(X_test, y_test))
# <<show score here>>
# Complete example using the Iris dataset:

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

iris = load_iris()

X = iris.data
y = iris.target

test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[ indices[:-test_size]]
y_train = y[ indices[:-test_size]]
X_test = X[ indices[-test_size:]]
y_test = y[ indices[-test_size:]]

import pandas as pd
y_train_ohe = pd.get_dummies(y_train)

# Instantiate a HyperoptEstimator with the search space and number of evaluations
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

import lightgbm as lgb

model_lgb = lgb.LGBMClassifier()
model_lgb.fit(X_train, y_train)
estim = HyperoptEstimator(classifier=model_lgb,
                          preprocessing=any_preprocessing('standard_scaler'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)

# Search the hyperparameter space based on the data

estim.fit( X_train, y_train, random_state=50 )
# estim.fit(X_train, y_train_ohe)

# Show the results

print( estim.score( X_test, y_test ) )
# 1.0

print( estim.best_model() )
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#           max_depth=3, max_features='log2', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=13, n_jobs=1,
#           oob_score=False, random_state=1, verbose=False,
#           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}