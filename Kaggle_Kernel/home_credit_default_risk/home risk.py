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

app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_REPCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
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

predict = log_reg.predict(train)
from sklearn.metrics import confusion_matrix
confusion_matrix(app_train["TARGET"], predict)

# performance of Logistic
data = app_train.copy()
from sklearn.cross_validation import train_test_split
data_train, data_test = train_test_split(data, test_size = 0.2, random_state = 42)
data_train_y = data_train['TARGET']
data_tran_x = data_train.drop(['TARGET'])