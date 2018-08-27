import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

bureau = pd.read_csv("/home/hungery/Documents/Kaggle_competion/home_credit/bureau.csv")
bureau.head()
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU':'previous_loan_counts'})

train = pd.read_csv("/home/hungery/Documents/Kaggle_competion/home_credit/application_train.csv")
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

# make a function to instead of above works
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