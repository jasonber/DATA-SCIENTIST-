import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
# suppress warings
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/"))

app_train = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_train.csv")
print('Training data shape:', app_train.shape)
app_train.head()

app_test = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_test.csv")
print('Testing data shape:', app_test.shape)
app_test.head()

app_train['TARGET'].value_counts()

app_train['TARGET'].astype(int).plot.hist()

def missing_values_table(df):

    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * df.isnull().sum() / len(df) 
         
    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
    
    mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})
       
    
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
        
    
    print(" Your selected dataframe has" + str(df.shape[1]) 
    + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0])
    + " columns that have missing values.")
     
    return mis_val_table_ren_columns

missing_values = missing_values_table(app_train)
missing_values.head(20)


app_train.dtypes.value_counts()

app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

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

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_train['TARGET'] = train_labels

print("Training Feature shape", app_train.shape)
print("Testing Feature shape", app_test.shape)

(app_train['DAYS_BIRTH'] / -365).describe()

app_train['DAYS_EMPLOYED'].describe()

app_train['DAYS_EMPLOYED'].plot.hist(title = "Days Employment Histogram")
plt.xlabel('Days Employment')

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]


print('The non-anomalies default on {:.2%} of loans'.format(non_anom['TARGET'].mean()))

print('The anomalies default on {:.2%} of loans'.format(anom['TARGET'].mean()))

print('There are {} anomalous days of employment'.format(len(anom)))

app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = "Days Employment Hsitogram")
plt.xlabel("Days Employment")

app_test["DAYS_EMPLOYED_ANOM"] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
print("There are {0:d} anomalies in the test data out of {1:d} entries".format(app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

correlations = app_train.corr()["TARGET"].sort_values()

print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMOst Negative Correlations:\n', correlations.head(15))

app_train["DAYS_BIRTH"] = abs(app_train['DAYS_BIRTH'])
app_train["DAYS_BIRTH"].corr(app_train['TARGET'])

plt.style.use('fivethirtyeight')
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');

plt.figure(figsize = (10, 8))
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');

age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num =11))
age_data.head(10)

age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups

plt.figure(figsize = (8, 8))
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');

ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs

plt.figure(figsize = (8, 6))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')

plt.figure(figsize = (10 ,12))
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    plt.subplot(3, 1, i + 1)
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    plt.title('Distribution of {} by Target Values'.format(source))
    plt.xlabel('{}'.format(source)); plt.ylabel('Density')

plt.tight_layout(h_pad = 2.5)

# plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()
# plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']
# plot_data = plot_data.dropna().loc[:100000, :]
#
# def corr_func(x, y, **kwargs):
#     r = np.corrcoef(x, y)[0][1]
#     ax = plt.gca()
#     ax.annotate('r = {:.2f}'.format(r),
#                 xy=(.2, .8), xycoords=ax.transAxes,
#                 size = 20)
#
# grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
#                     hue = 'TARGET',
#                     vars = [x for x in list(plot_data.columns) if x != 'TARGET'])
#
# grid.map_upper(plt.scatter, alpha = 0.2)
# grid.map_diag(sns.kdeplot)
# grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);
#
# plt.suptitle('Ext Source and Age Featur Paris Plot', size = 32, y = 1.05);
#
#

# polynomial feature
poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree = 3)

poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Feature shape:', poly_features.shape)

poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                         'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features["TARGET"] = poly_target
poly_corrs = poly_features.corr()["TARGET"].sort_values()
print(poly_corrs.head(10))
print(poly_corrs.tail(5))

poly_features_test = pd.DataFrame(poly_features_test, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1',
                                                                                                    'EXT_SOURCE_2',
                                                                                                    'EXT_SOURCE_3',
                                                                                                    'DAYS_BIRTH']))

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = "SK_ID_CURR", how = 'left')

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape: ', app_test_poly.shape)

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

plt.figure(figsize = (12, 20))
for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    plt.subplot(4, 1, i + 1)
    sns.kdeplot(app_train_domain.loc[app_train['TARGET'] == 0, feature], label = 'target == 0')
    sns.kdeplot(app_train_domain.loc[app_train['TARGET'] == 1, feature], label = 'target == 1')
    plt.title("Distribution of {} by TARGET Value".format(feature))
    plt.xlabel("{}".format(feature)); plt.ylabel('Density');
plt.tight_layout(h_pad = 2.5)

from sklearn.preprocessing import MinMaxScaler, Imputer
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()

features = list(train.columns)

test = app_test.copy()
imputer = Imputer(strategy = 'median')
scaler = MinMaxScaler(feature_range = (0, 1))

imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(train, train_labels)

log_reg_pred = log_reg.predict_proba(test)[:, 1]
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()


# model selection
X = train.copy()
y = train_labels.copy()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(C = 0.0001, class_weight = 'balanced')
classifier_LR.fit(X_train, y_train)

def model_score(model, X = X_test, y = y_test):
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
    return print('The score of {} is {}'.format(model, np.sqrt(-score).mean()))


# LR_score = cross_val_score(classifier_LR, X_train, y_train, scoring = 'neg_mean_squared_', cv = 10)
#
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')
classifier_RF.fit(X_train, y_train)

model_score(classifier_LR)
model_score(classifier_RF)

# confusion matrix
LR_y = classifier_LR.predict(X_test)
RF_y = classifier_RF.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, LR_y)
confusion_matrix(y_test, RF_y)

from sklearn.metrics import precision_score, recall_score
precision_score(y_test, LR_y)
recall_score(y_test, LR_y)

precision_score(y_test, RF_y)
recall_score(y_test,RF_y)

from sklearn.metrics import f1_score
f1_score(y_test, RF_y)
f1_score(y_test, LR_y)

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, RF_y)
precision_LR, recall_LR, threshold_LR = precision_recall_curve(y_test, LR_y)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

plt.plot(precision_LR, recall_LR, label = 'LR')
plt.xlabel('Recalls')
plt.ylabel('Precision')
plt.plot(precisions, recalls, label ='RF')
plt.legend()
plt.show()

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, RF_y)
fpr_LR, tpr_LR, threshold_LR = roc_curve(y_test, LR_y)
plt.plot(fpr, tpr,label = "RF")
plt.plot(fpr_LR, tpr_LR, label = "LR")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# # imbalance smaples
# from imblearn.over_sampling import SMOTE
# model_smote = SMOTE()
# X_somte, y_somte = model_smote.fit_sample(X, y)

# LGBM
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    labels = features['TARGET']

    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])

    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        fetaures, test_features = features.align(test_features, join = 'inner', axis = 1)
        cat_indeices = 'auto'

    elif encoding == 'le':
        label_encoder = LabelEncoder()
        cat_indeices = []
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                cat_indeices.append(i)

    else:
        raise ValueError("Encoding must be either 'ohe' or 'le' ")

    print('Trianing Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)

    features = np.array(features)
    test_features = np.array(test_features)

    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state =50)
    feature_importance_value = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])

    valid_score = []
    train_score = []

    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        model = lgb.LGBMClassifier(n_estimators = 10000, objective = 'binary',
                                   class_weight = 'balanced', learning_rate = 0.05,
                                   reg_alpha = 0.1, reg_lambda = 0.1,
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indeices,
                  early_stopping_rounds = 100, verbose = 200)

        best_iteration = model.best_iteration_

        feature_importance_value += model.feature_importances_ / k_fold.n_splits

        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_roba(valid_features, num_iteration = best_iteration)[:, 1]

        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_score.append(valid_score)
        train_score.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})

    valid_auc = roc_auc_score(labels, out_of_fold)

    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_score))

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_score,
                            'valid': valid_score})

    return submission, feature_importances, metrics

submission, fi, metrics = model(app_train, app_test)
print('Baseline metrics')
print(metrics)

# domain knowledge feature
app_train_domain['TARGET'] = train_labels

submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)