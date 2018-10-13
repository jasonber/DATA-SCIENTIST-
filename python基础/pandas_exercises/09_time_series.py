# Apple Stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
path = '/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/09_Time_Series/Apple_Stock/appl_1980_2014.csv'
apple = pd.read_csv(path)

# 查看数据类型
apple.dtypes

# 将日期转换为datetime
apple.Date = pd.to_datetime(apple.Date)

# 将日期设置为索引
apple.set_index(apple.Date, inplace=True)

# 重复数据
apple.index.is_unique

# 日期从最早的开始排序
apple.sort_index(inplace=True)

# 找到每月的最晚交易时间
apple_month = apple.resample('BM').mean()

# 第一天与最后一天相差多少天
(apple.index.max() - apple.index.min()).days

# 数据中有多少个月
apple_months = apple.resample('BM').mean()
len(apple_months)

# 画出Adj Close的图，13.5*9大小
plt.figure(figsize=(13.5, 9))
plt.plot(apple.loc[:, 'Adj Close'])
plt.ylabel("Adj Close")
plt.xlabel('Year')
plt.title('Apple Stock')

# Getting Financial Data
import pandas as pd
import datetime as dt
from pandas_datareader import data as web

# 创建你的时间范围， 从2015/01/01到今天
start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime.today()

# 选择apple， tesla, IBM, Linkedin 的股票数据
stocks = ['AAPL', 'TSLA', 'IBM', 'MSFT']

# 从谷歌上获取数据
df = web.DataReader(stocks, 'yahoo', start_date, end_date)

# df的类型
type(df)

# df有哪些属性
df.columns

# 创建一个volume values的vol数据框
vol = df.loc[:, 'Volume']

# 按周分类，切记还要考虑年份
vol['week'] = vol.index.week
vol['year'] = vol.index.year
week = vol.groupby(['year', 'week']).sum()

# 查找2015年的数据
year = vol.drop(columns= 'week')
year = year[year['year'] == 2015].sum()

# Investor_FlowJ_of_funds_Us
import pandas as pd

# 载入数据
url = "https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv"
df = pd.read_csv(url)

# 数据的频率是什么。。。。。。
# 将date作为索引
df = df.set_index(df['Date'])

# index的类型
df.index

# 索引变成datetime格式
df.index = pd.to_datetime(df.index)
# 频率改为月份
monthly = df.resample("M").sum()
# 删除空值
monthly = monthly.dropna()
# 把月份频次转换为年份
year = monthly.resample("AS-JAN").sum()