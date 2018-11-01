# the tutorial of featuretools
# this toolkit is ideal tool for solving the problem of several related tables to combined into a dataframe

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import featuretools as ft


# get the full data
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

# get the small data
key1 = ['SK_ID_CURR']
key2 = key1.copy()
key2.extend(['SK_ID_PREV'])
app_train = app_train.sort_values(key1).reset_index(drop=True).loc[:1000, :]
app_test = app_test.sort_values(key1).reset_index(drop=True).loc[:1000, :]
bureau = bureau.sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop=True).loc[:1000, :]
bureau_balance = bureau_balance.sort_values(['SK_ID_BUREAU']).reset_index(drop=True).loc[:1000, :]
cash = cash.sort_values(key2).reset_index(drop=True).loc[:1000, :]
credit = credit.sort_values(key2).reset_index(drop=True).loc[:1000, :]
previous = previous.sort_values(key2).reset_index(drop=True).loc[:1000, :]
installments = installments.sort_values(key2).reset_index(drop=True).loc[:1000, :]

# add indentify column
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test['TARGET'] = np.nan
app = app_train.append(app_test, ignore_index=True)

# entity set with id applications
es = ft.EntitySet(id='clients')

# entities with a unique index
es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')

# entity tha do not have a unique index
es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance,
                              make_index=True, index='bureaubalance_index')
es = es.entity_from_dataframe(entity_id='cash', dataframe=cash, make_index=True, index='cash_index')
es = es.entity_from_dataframe(entity_id='installments', dataframe=installments, make_index=True,
                              index='installments_index')
es = es.entity_from_dataframe(entity_id='credit', dataframe=credit, make_index=True, index='credit_index')

# make the relationship
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# add in the defined relationship
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous, r_previous_cash, r_previous_installments,
                           r_previous_credit])
