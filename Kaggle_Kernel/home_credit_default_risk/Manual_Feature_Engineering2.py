import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

plt.style.use('fivethirtyeight')

def agg_numerical(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.

    Parameters
    --------
        df (dataframe):
            the child dataframe to calculate the statistics on
        parent_var (string):
            the parent variable used for grouping and aggregating
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated by the `parent_var` for
            all numeric columns. Each observation of the parent variable will have
            one row in the dataframe with the parent variable as the index.
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed.

    """
    for col in df:
        if col != parent_var and "SK_ID" in col:
            df = df.drop(columns = col)

    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    columns = []
    for var in agg.columns.levels[0]:
        if var != parent_var:
            for stat in agg.columns.levels[1]:
                columns.append("{}_{}_{}".format(df_name, var, stat))

    agg.columns = columns

    _, idx = np.unique(agg, axis = 1, return_index = True)
    agg = agg.iloc[:, idx]

    return agg

def agg_categorical(df, parent_var, df_name):
    """
        Aggregates the categorical features in a child dataframe
        for each observation of the parent variable.

        Parameters
        --------
        df : dataframe
            The dataframe to calculate the value counts for.

        parent_var : string
            The variable by which to group and aggregate the dataframe. For each unique
            value of this variable, the final dataframe will have one row

        df_name : string
            Variable added to the front of column names to keep track of columns


        Return
        --------
        categorical : dataframe
            A dataframe with aggregated statistics for each observation of the parent_var
            The columns are also renamed and columns with duplicate values are removed.

        """
    categorical = pd.get_dummies(df.select_dtypes('category'))
    categorical[parent_var] = df[parent_var]
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])

    column_names = []

    for var in categorical.columns.levels[0]:
        for stat in ['sum', 'count', 'mean']:
            column_names.append('{}_{}_{}'.format(df_name, var, stat))

    categorical.columns = column_names

    _,idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]

    return categorical

def kde_target(var_name, df):
    corr = df["TARGET"].corr(df[var_name])
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize = (12, 6))

    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    plt.xlabel(var_name); plt.ylabel('Density');plt.title("{} Distribution".format(var_name))
    plt.legend();

    print('The correlation between {} and the TARGET is {:.4f}'.format(var_name, corr))
    print('Median value for loan that was not repaid = {:.4f}'.format(avg_not_repaid))
    print('Median value for loan that was repaid = {:.4f}'.format(avg_repaid))

# convert data type
import sys
def return_size(df):
    """return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    original_memory = df.memory_usage().sum()
    for c in df:
        if ("SK_ID" in c):
            df[c] = df[c].fillna(0).astype(np.int32)
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')

        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)

    new_memory = df.memory_usage().sum()

    if print_info:
        print('Original Memory Usage: {} gb.'.format(round(original_memory / 1e9, 2)))
        print('New Memory Usage: {} gb.'.format(round(new_memory / 1e9, 2)))

    return df

previous = pd.read_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/previous_application.csv')
previous = convert_types(previous, print_info = True)
previous_agg = agg_numerical(previous, "SK_ID_CURR", 'previous')
print('Previous aggregation shape:{}'.format(previous_agg.shape))
previous_counts = agg_categorical(previous, "SK_ID_CURR", 'previous')
print('Previous counts shape: ', previous_counts.shape)

train = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_train.csv")
train = convert_types(train)
test = pd.read_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_test.csv')
test = convert_types(test)

train = pd.merge(train, previous_counts, on = "SK_ID_CURR", how = "left")
train = pd.merge(train, previous_agg, on = "SK_ID_CURR", how = "left")

test = pd.merge(test, previous_counts, on = "SK_ID_CURR", how = 'left')
test = pd.merge(test, previous_agg, on = "SK_ID_CURR", how = "left")

gc.enable()
del previous, previous_agg, previous_counts
gc.collect()

def missing_values_table(df, print_info = False):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0: 'Missing Values', 1:'% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending = False).round(1)

    if print_info:
        print("Your selected dataframe has" + str(df.shape[1]) + "columns.\n""There are " +
              str(mis_val_table_ren_columns.shape[0])+
              "columns that have missing values.")

    return mis_val_table_ren_columns

def remove_missing_columns(train, test, threshold = 90):
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)

    test_miss = pd.DataFrame(test.isnull().sum())
    # print("test_miss 1", test_miss)
    test_miss['percent'] = 100 * test_miss[0] / len(test)

    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])
    # print(train_miss)
    # print(test_miss)

    missing_columns = list(set(missing_train_columns + missing_test_columns))
    # print(missing_test_columns)

    print('There are {} columns with greater than {}% missing values.'.format(len(missing_columns), threshold))

    train = train.drop(columns = missing_columns)
    test = test.drop(columns = missing_columns)

    return train, test

train, test = remove_missing_columns(train, test)
train.to_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/features_engineer/train_FE2_missing_remove.csv', index = False)
test.to_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/features_engineer/test_FE2_missing_remove.csv', index = False)

# gc.enable()
# del test, train
# gc.collect()


def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level
       at the client level

       Args:
           df (dataframe): data at the loan level
           group_vars (list of two strings): grouping variables for the loan
           and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
           names (list of two strings): names to call the resulting columns
           (example ['cash', 'client'])

       Returns:
           df_client (dataframe): aggregated numeric stats at the client level.
           Each client will have a single row with all the numeric data aggregated
       """
    df_agg = agg_numerical(df, parent_var = group_vars[0], df_name = df_names[0])
    if any(df.dtypes == 'category'):
        df_counts = agg_categorical(df, parent_var = group_vars[0], df_name = df_names[0])
        df_by_loan = pd.merge(df_counts, df_agg, on = group_vars[0], how = 'outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        df_by_loan = pd.merge(df_by_loan, df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])
        df_by_client = agg_numerical(df_by_loan, parent_var = group_vars[1], df_name = df_names[1])

    else:
        df_by_loan = pd.merge(df_agg, df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')

        gc.enable()
        del df_agg
        gc.collect()

        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])

        df_by_client = agg_numerical(df_by_loan, parent_var = group_vars[1], df_name = df_names[1])

    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client

cash = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/POS_CASH_balance.csv")
cash = convert_types(cash, print_info = True)

cash_by_client = aggregate_client(cash, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['cash', 'client'])

print('Cash by Client Shape:', cash_by_client.shape)
train = pd.merge(train, cash_by_client, on='SK_ID_CURR', how='left')
test = pd.merge(test, cash_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del cash, cash_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

credit = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/credit_card_balance.csv")
credit = convert_types(credit, print_info=True)
credit_by_client = aggregate_client(credit, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['credit', 'client'])
print('Credit by client shape:', credit_by_client.shape)
train = pd.merge(train, credit_by_client, on='SK_ID_CURR', how='left')
test = pd.merge(test, credit_by_client, on='SK_ID_CURR', how='left')
gc.enable()
del credit, credit_by_client
gc.collect()
train, test = remove_missing_columns(train, test)

# Installment
installment = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/credit_card_balance.csv")
installment = convert_types(installment, print_info=True)
installment_by_client = aggregate_client(installment, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['installments',
                                                                                                         'client'])
print('Installment by client shape:', installment_by_client.shape)
train = pd.merge(train, installment_by_client, on='SK_ID_CURR', how='left')
test = pd.merge(test, installment_by_client, on='SK_ID_CURR', how='left')
gc.enable()
del installment, installment_by_client
gc.collect()
train, test = remove_missing_columns(train, test)

print('Final Training Shape:', train.shape)
print('Final Testing Shape:', test.shape)
print('Final Training Size:{}'.format(return_size(train)))
print('Final Testing Size:{}'.format(return_size(test)))

# 保存数据集
train.to_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/train_previous_raw.csv", index=False, chunksize=500)
test.to_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/test_previous_raw.csv", index=False)