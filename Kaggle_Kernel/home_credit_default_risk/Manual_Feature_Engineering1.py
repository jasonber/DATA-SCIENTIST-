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

# drop the collinear variable
# if the correlation of pair variables >0.8, we see them as collinear
# key表示一个变量，value表示与key共线且>0.8的变量。所以与key不同的value就是共线性变量
threshold =0.8
above_threshold_vars = {}
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

for key, value in above_threshold_vars.items():
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)

cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))

train_corrs_removed = train.drop(columns = cols_to_remove)
test_corrs_removed = test.drop(columns = cols_to_remove)

print("Training Corrs Removed Shape: ", train_corrs_removed.shape)
print("Testing Corrs Removed Shape: ", test_corrs_removed.shape)

train_corrs_removed.to_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/train_bureau_corrs_removed.csv', index = False)
test_corrs_removed.to_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/test_bureau_corrs_removed.csv', index = False)

# free up memory
import gc
gc.enable()
del bureau_balance_by_client, bureau_counts, above_threshold_vars, categorical, cols_seen, cols_to_remove, cols_to_remove_pair,
corr, corrs, missing_columns, missing_test, missing_test_vars, missing_train, missing_train_vars, test, train_corrs_removed,
missing_train, train_corrs_removed, test_corrs_removed
gc.collect()

# model and model seletion
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
train_corrs_removed = pd.read_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/train_bureau_corrs_removed.csv')
test_corrs_removed = pd.read_csv('/home/zhangzhiliang/Documents/Kaggle_data/home_risk/test_bureau_corrs_removed.csv')

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    """Train and test a light gradient boosting model using
        cross validation.

        Parameters
        --------
            features (pd.DataFrame):
                dataframe of training features to use
                for training a model. Must include the TARGET column.
            test_features (pd.DataFrame):
                dataframe of testing features to use
                for making predictions with the model.
            encoding (str, default = 'ohe'):
                method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
                n_folds (int, default = 5): number of folds to use for cross validation

        Return
        --------
            submission (pd.DataFrame):
                dataframe with `SK_ID_CURR` and `TARGET` probabilities
                predicted by the model.
            feature_importances (pd.DataFrame):
                dataframe with the feature importances from the model.
            valid_metrics (pd.DataFrame):
                dataframe with training and validation metrics (ROC AUC) for each fold and overall.

        """
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    labels = features['TARGET']
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    print("train original: {}\ntest original: {}".format(features.shape, test_features.shape))

    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        cat_indices = 'auto'
        print("train dummies: {}\ntest dummies: {}".format(features.shape, test_features.shape))

    elif encodeing == 'le':
        label_encoder = LabelEncoder()
        cat_indices = []

        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                cat_indices.append(i)

    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)
    features = np.array(features)
    test_features = np.array(test_features)
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    valid_scores = []
    train_scores = []

    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', class_weight = 'balanced',
                                   learning_rate = 0.05, reg_alpha = 0.1, reg_lambda = 0.1, subsample = 0.8,
                                   n_jobs = -1, random_state = 50)

        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 10)

        best_iteration = model.best_iteration_

        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:,1]

        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance':feature_importance_values})

    valid_auc = roc_auc_score(labels, out_of_fold)
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    print('train {}, valid {}'.format(train_scores, valid_scores))

    metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores})

    return submission, feature_importance, metrics

# plot the importance of features
def plot_feature_importances(df):
    """
       Plot importances returned by a model. This can work with any measure of
       feature importance provided that higher importance is better.

       Args:
           df (dataframe): feature importances. Must have the features in a column
           called `features` and the importances in a column called `importance

       Returns:
           shows a plot of the 15 most importance features

           df (dataframe): feature importances sorted by importance (highest to lowest)
           with a column for normalized importance
           """

    df = df.sort_values('importance', ascending = False).reset_index()
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    plt.figure(figsize = (10, 6))
    ax = plt.subplot()

    ax.barh(list(reversed(list(df.index[:15]))), df['importance_normalized'].head(15), align = 'center', edgecolor = 'k')
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()

    return df

submission_corrs, fi_corrs, metrics_corr = model(train_corrs_removed, test_corrs_removed)

fi_corrs_sorted = plot_feature_importances(fi_corrs)

submission_corrs.to_csv('test_two.csv', index = False)