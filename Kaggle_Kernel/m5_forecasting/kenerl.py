import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np 

data_path = "/home/zhang/Documents/data_set/m5_forecasting_accuracy/"

calendar = pd.read_csv(f'{data_path}/calendar.csv')
sell_prices = pd.read_csv(f'{data_path}/sell_prices.csv')
sample_submission = pd.read_csv(f'{data_path}/sample_submission.csv')
sales_train_val = pd.read_csv(f'{data_path}/sales_train_validation.csv')

# sample sales data
ids = sorted(list(set(sales_train_val['id'])))
d_cols = [c for c in sales_train_val.columns if 'd_' in c]

x_1 = sales_train_val.loc[sales_train_val['id'] == ids[2]].set_index('id')[d_cols].values[0]
x_2 = sales_train_val.loc[sales_train_val['id'] == ids[1]].set_index('id')[d_cols].values[0]
x_3 = sales_train_val.loc[sales_train_val['id'] == ids[17]].set_index('id')[d_cols].values[0]

fig = plt.figure()
subplot1 = fig.add_subplot('311')
subplot1.hist(x=np.arange(len(x_1)), y=x_1, color='green')
subplot2 = fig.add_subplot('312')
subplot2.hist(x=np.arange(len(x_2)), y=x_2, color='violet')
subplot3 = fig.add_subplot('313')
subplot3.hist(x=np.arange(len(x_3)), y=x_3, color='blue')
fig.show()
