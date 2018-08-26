import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
# suppress warings
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("/home/hungery/Documents/Kaggle_competion/home_credit/"))

app_train = pd.read_csv("/home/hungery/Documents/Kaggle_competion/home_credit/application_train.csv")
print('Training data shape:', app_train.shape)
app_train.head()

app_test = pd.read_csv("/home/hungery/Documents/Kaggle_competion/home_credit/application_test.csv")
print('Testing data shape:', app_test.shape)
app_test.head()

app_train['TARGET'].value_counts()

app_train['TARGET'].astype(int).plot.hist()


def missing_values_table(df):
    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    print(" Your selected dataframe has" + str(df.shape[1])
          + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0])
          + " columns that have missing values.")

    return mis_val_table_ren_columns


missing_values = missing_values_table(app_train)
missing_values.head(20)

app_train.dtypes.value_counts()

app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

le = LabelEncoder()
le_count = 0

for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            le_count += 1
print("{} columns were label encode.".format(le_count))

app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Feature shape', app_train.shape)
print('Test Feature shape', app_test.shape)

train_labels = app_train["TARGET"]

app_train, app_test = app_train.align(app_test, join='inner', axis=1)
app_train['TARGET'] = train_labels

# domain knowledge
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

app_train_domain['TARGET'] = train_labels

import lightgbm as lgb
X = app_train_domain.copy().drop(columns = ['TARGET', 'SK_ID_CURR'])
y = app_train_domain.copy()['TARGET'].values
y = pd.DataFrame({'TARGET':y})

predict_X = app_test_domain.copy()
predict_X = predict_X.drop(columns  = ['SK_ID_CURR'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


model_domain =lgb.LGBMClassifier(n_estimators=10000, objective='binary', class_weight='balanced', learning_rate=0.05,
                                 reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, n_jobs=-1, random_state=50)

from hyperopt import tpe
from hpsklearn import HyperoptEstimator, any_preprocessing
# X_train_reindex = X_train.copy().reset_index(drop = True)
# y_train_reindex = y_train.copy().reset_index(drop = True)


# estim = HyperoptEstimator(classifier=model_domain, preprocessing=any_preprocessing('one_hot_encoder'), algo=tpe.suggest,
#                           max_evals=100, trial_timeout=120)
# estim.fit(X_train, y_train, random_state=50)
# estim.fit()

# don`t use hpsklearn
from hyperopt import hp, fmin, rand, tpe, space_eval,
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

def model_tune (args):
    model_domain = lgb.LGBMClassifier(n_estimators=10000, objective='binary', class_weight='balanced',
                                      learning_rate=0.05,
                                      reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, n_jobs=-1, random_state=50)
    model_domain.fit(X_train, y_train, verbose=3)
    predict = model_domain.predict(X_test)
    score = accuracy_score(y_test, predict)
    print(score)
    return -score


space = {'n_estimators':hp.uniform('n_estimators', 0, 20000),
         'learning_rate':hp.uniform('learning_rate', 0, 1),
         'reg_alpha':hp.uniform('reg_alpha', 0, 1),
         'reg_lambda':hp.uniform('reg_lambda', 0, 1),
         'subsample':hp.uniform('subsample', 0, 1)
         }
best = fmin(model_tune, space, algo=tpe.suggest, max_evals=3)

from sklearn.externals import joblib
joblib.dump(best, "LGB_hp.pkl")
LGB_hp = joblib.load('LGB_hp.pkl')


best.
y_pred_te = best.predict(X_test)
print(best.)





print(space_eval(space, best))

model_best = lgb.LGBMClassifier(n_estimators=1655, objective='binary', class_weight='balanced',
                                  learning_rate=0.49637306466316566,
                                  reg_alpha= 0.2180451659755328, reg_lambda= 0.4496755142995127,
                                subsample=0.02267574289056462, n_jobs=-1, random_state=50)

model_domain =lgb.LGBMClassifier(n_estimators=10000, objective='binary', class_weight='balanced', learning_rate=0.05,
                                 reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, n_jobs=-1, random_state=50)


model_best.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_test, y_test), (X_train, y_train)],
                 eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=5)
best_y = model_best.predict(X_test)

model_domain.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_test, y_test), (X_train, y_train)],
                 eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=5)
domain_y = model_domain.predict(X_test)

from sklearn.metrics import accuracy_score
score_best = accuracy_score(y_test, best_y)
score_domain = accuracy_score(y_test, domain_y)

print('auc of best model:{:.2f} \nauc of domain model:{:.2f} \n'.format(score_best, score_domain))