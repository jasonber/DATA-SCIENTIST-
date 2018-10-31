# the tutorial of featuretools
# this toolkit is ideal tool for solving the problem of several related tables to combined into a dataframe

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import featuretools

# get data
def get_data(path, data_name):
    create_vars[data_name] = pd.read_csv(path)
    print("Success")
    return create_vars[data_name]


path = "/home/zhangzhiliang/Documents/Kaggle_data/home_risk/"
file_list = ['application_train', 'application_test', 'bureau', 'bureau_balance', 'POS_CASH_balance',
             'credit_card_balance', 'previous_application', 'installments_payments']
df_name = ['app_train', 'app_test', 'bureau', 'bureau_balance', 'cash',
           'credit', 'previous', 'installments']

path_list = []
for i in file_list:
    path_list.append(path + i + '.csv')

create_vars = locals()
for i, j in zip(path_list, df_name):
    create_vars[j] = get_data(i, j)

# add indentify column
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test['TARGET'] = np.nan

app = pd.merge(app_train, app_test, ignore_index=True)
