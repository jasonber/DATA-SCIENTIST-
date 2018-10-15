# Iris
import pandas as pd
import numpy as np

# 载入数据
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(url)

# 创建列
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 空值查询
iris.isnull().sum()

# 将‘petal_length'的10到29行设置为空值
iris.iloc[10:30, 2] = np.nan

# 将空值设为1
iris.fillna(1, inplace=True)

# 删除class列
iris.drop(columns='class', inplace=True)

# 将前3行设为空值
iris.loc[0:2, :] = np.nan

# 删除有空值的行
iris.dropna(axis=0, how='any', inplace=True)

# 重置index，从0 开始
iris = iris.reset_index(drop=True)

# Wine
import pandas as pd
import numpy as np

# 载入数据
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine = pd.read_csv(url)

# 删除1 4 7 9 13 14
wine = wine.drop(columns=wine.columns[[0, 3, 6, 8, 12, 13, 11]])

# 设置列名
wine.columns = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'magnesium', 'flavanolds', 'proanthocyanins', 'hue']

# 将alcohol的前三行设置为Na
wine.loc[0:2, 'alcohol'] = np.nan

# 将magnesium的第3行到第4行设为Nan
wine.loc[2:3, 'magnesium'] = np.nan

# 将alcohol的空值设置为10 ，magnesium的空值设为100
wine['alcohol'].fillna(10, inplace=True)
wine['magnesium'].fillna(100, inplace=True)

# 查看空值数目
wine.isnull().sum()

# 创建一个数组。数组中有10个0到10的随机数
random = np.random.randint(10, size=10)

# 随机赋予空值
wine.loc[random, 'alcohol'] = np.nan

# 缺失值有多少
wine.isnull().sum()

# 删除空值行
wine = wine.dropna(axis=0, how='any')

# 输出alcohol的非空值
mask = wine['alcohol'].notnull()
wine.alcohol[mask]

# 重置索引
wine.reset_index(drop=True, inplace=True)