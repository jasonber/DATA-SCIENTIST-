# !/usr/bin/python
# -*-coding:utf-8-*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint

# 手写读取数据
# if __name__ == "__main__":
#     path = '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/data/Advertising.csv'
#     f = open(path, mode='r')
#     x = []
#     y = []
#     for i, d in enumerate(f):
#         if i == 0: # 跳过列名那一行
#             continue
#         d = d.strip() # 去掉换行、回车等
#         if not d:
#             continue
#         d = list(map(float, d.split(',')))
#         x.append(d[1:-1])
#         y.append(d[:-1])
#     pprint(x)
#     pprint(y)
#     x = np.array(x)
#     y = np.array(y)
# f.close()
#
# # python 自带库打开数据
# path = '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/data/Advertising.csv'
# f = open(path, mode='r')
# data = []
# print(f)
# d = csv.reader(f)
# for line in d:
#     data.append(line)
#     print(line)
# f.close()

# pandas
path = '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/data/Advertising.csv'
data = pd.read_csv(path)
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
