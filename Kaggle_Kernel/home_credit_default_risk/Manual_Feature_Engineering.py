import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

bureau = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/bureau.csv")
bureau.head()
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU':'previous_loan_counts'})

# make a function to instead of above works
train = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_train.csv")
train = pd.merge(train, previous_loan_counts, on="SK_ID_CURR", how='left')
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)

def kde_target(var_name, df):
    corr = df['TARGET'].corr(df[var_name])
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize = (12, 6))

    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    plt.xlabel(var_name);plt.ylabel('Density');plt.title('{}Distribution'.format(var_name));plt.legend();

    print('The correlation between {} and the TARGET is {:.4f}'.format(var_name, corr))
    print('Median value for loan that was not repaid = {:.4f}'.format(avg_not_repaid))
    print('Median value for loan that was repaid = {:.4f}'.format(avg_repaid))

kde_target('EXT_SOURCE_3', train)
kde_target('previous_loan_counts', train)

# explorate the numeric feature and merge application.csv with bureau.csv
bureau_agg = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max',
                                                                                                  'min', 'sum']).reset_index()
columns = ['SK_ID_CURR']

for var in bureau_agg.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in bureau_agg.columns.levels[1][:-1]:
            columns.append('bureau_{}_{}'.format(var, stat))

bureau_agg.columns = columns

train = pd.merge(train, bureau_agg, on = 'SK_ID_CURR', how = 'left')

new_corrs = []
for col in columns:
    corr = train['TARGET'].corr(train[col])
    new_corrs.append((col, corr))

new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)

kde_target('bureau_DAYS_CREDIT_mean', train)

def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.

    Parameters
    --------
        df (dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to group df
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated for
            all numeric columns. Each instance of the grouping variable will have
            the statistics (mean, min, max, sum; currently supported) calculated.
            The columns are also renamed to keep track of features created."""
    for col in df:
        if col != group_var and "SK_ID" in col:
            df = df.drop(columns = col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    columns = [group_var]

    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                columns.append('{}_{}_{}'.format(df_name, var, stat))
    agg.columns = columns
    return agg

bureau_agg_new = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')

# function of correlations
def target_corrs(df):
    corrs = []

    for col in df.columns:
        print(col)
        if col != 'TARGET':
            corr = df["TARGET"].corr(df[col])
            corrs.append((col, corr))

    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    return corrs

# explorate the category features
categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum', 'mean'])

# from multi-level columns to one-level
group_var = 'SK_ID_CURR'
columns = []
for var in categorical_grouped.columns.levels[0]:
    if var != group_var:
        for stat in ['count', 'count_norm']:
            # make the new name
            columns.append('{}_{}'.format(var, stat))

categorical_grouped.columns = columns

train = pd.merge(train, categorical_grouped, on = 'SK_ID_CURR', how = 'left', right_index = True)

# make a function to replace above codes
def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
       of `group_var` of each unique category in every categorical variable

       Parameters
       --------
       df : dataframe
           The dataframe to calculate the value counts for.

       group_var : string
           The variable by which to group the dataframe. For each unique
           value of this variable, the final dataframe will have one row

       df_name : string
           Variable added to the front of column names to keep track of columns


       Return
       --------
       categorical : dataframe
           A dataframe with counts and normalized counts of each unique category in every categorical variable
           with one row for every unique value of the `group_var`.

       """
    categorical = pd.get_dummies(df.select_dtypes('object'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    columns_names = []
    for var in categorical.columns.levels[0]:
        for stat in ['count', 'count_norm']:
            columns_names.append("{}_{}_{}".format(df_name, var, stat))
    categorical.columns = columns_names

    return categorical

bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')

# handle the bureau_balance.csv
bureau_balance = pd.read_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/bureau_balance.csv')
# category features
bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
# numerical features
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
# merge counts and agg
bureau_by_loan = pd.merge(bureau_balance_agg, bureau_balance_counts, how = 'outer', on = 'SK_ID_BUREAU')
bureau_by_loan = pd.merge(bureau_by_loan, bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')

# get the features of loan with every clients by SK_ID_CURR and named client_XX_XX
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR',
                                       df_name = 'client')

# free up memory
import gc
gc.enable()
del train, bureau, bureau_balance, bureau_agg, bureau_agg_new, bureau_balance_agg, bureau_balance_counts, bureau_by_loan,
bureau_balance_by_client, bureau_counts
gc.collect()

# putting the functions together
train = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_train.csv")
bureau = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/bureau.csv")
bureau_balance = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/bureau_balance.csv")

bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')

bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

bureau_by_loan = pd.merge(bureau_balance_agg, bureau_balance_counts, right_index = True, how = 'outer',
                          on ='SK_ID_BUREAU')
bureau_by_loan = pd.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR',
                                       df_name = 'client')
# merge above table into train table
original_features = list(train.columns)
print('Original Number of Features: ', len(original_features))

train = pd.merge(train, bureau_counts, on = 'SK_ID_CURR', how = 'left')
train = pd.merge(train, bureau_agg, on = 'SK_ID_CURR', how = 'left')
train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

new_features = list(train.columns)
print('Number of features using previous loans from other institutions data: ', len(new_features))


# feature selction
# drop the columns with too many missing values
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0: 'Missing Values', 1: '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] !=0].sort_values(
        "% of Total Values", ascending = False).round(1)

    print("Your Selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values")
    return mis_val_table_ren_columns

missing_train = missing_values_table(train)

# find the features which the missing percent greater than 90%
# these features should be drop
missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])

# calculate information for testing data and merge the other institution information datas
test = pd.read_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/application_test.csv")
test = pd.merge(test, bureau_counts, on = 'SK_ID_CURR', how = 'left')
test = pd.merge(test, bureau_agg, on = 'SK_ID_CURR', how = 'left')
test = pd.merge(test, bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
print('Shape of Testing Data: {}'.format(test.shape))

# make the columns name same
train_labels = train['TARGET']
train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels

print('Training Data Shape: ', train.shape)
print('Testing Data Shape: ', test.shape)

missing_test = missing_values_table(test)
missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])
len(missing_test_vars)

missing_columns = list(set(missing_test_vars + missing_train_vars))
print("There are {:d} columns with more than 90% missing in either the train or testing data.".format(len(missing_columns)))

# drop the missing columns
train = train.drop(columns = missing_columns)
test = test.drop(columns = missing_columns)

train.to_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/train_bureau_raw.csv", index = False)
test.to_csv("/home/zhangzhiliang/Documents/Kaggle_data/home_risk/test_bureau_raw.csv", index = False)

corrs = train.corr()
corrs = corrs.sort_values('TARGET', ascending = False)
pd.DataFrame(corrs['TARGET'].head(10))
pd.DataFrame(corrs['TARGET'].dropna().tail(10))

kde_target(var_name = 'bureau_CREDIT_ACTIVE_Active_count_norm', df = train)

