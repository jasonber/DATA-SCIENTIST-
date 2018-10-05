import pandas as pd
import numpy as np

# chipotle 是一家主打墨西哥风味的美国快餐
# get the data
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep ='\t')
# see the first 10 entries
chipo.head(10)
# what is the number of observations in the dataset
# method 1 type(chipo.shape)
chipo.shape[0]
# method 2
chipo.info()
# 列数
chipo.shape[1]
# 列名
chipo.columns
# 索引
chipo.index

# 最多的物品
most_order = chipo.groupby('item_name').sum().sort_values(['quantity'], ascending=False)

# 有多少物品参与了排序
most_order.shape

# 在choice_description 中最多的描述
most_des = chipo.groupby('choice_description').sum().sort_values(['quantity'], ascending=False)

# 一共有多少题目
order_sum = chipo['quantity'].sum()

''' 将item_price调整为float '''
chipo.item_price.dtype
dollarizer = lambda x: float(x[1: -1])
chipo['item_price'] = chipo['item_price'].apply(dollarizer)
chipo.item_price.dtype

# 数据中的收益是多少
revenue = (chipo['quantity'] * chipo['item_price']).sum()
type(revenue)
print("Revenue is: ${}".format(np.round(revenue, 2)))

# 有多少订单
orders = chipo['order_id'].value_counts().count()
orders

# 每一笔订单的平均收益
chipo['revenus'] = chipo['quantity'] * chipo['item_price']
chipo.groupby(['order_id']).revenus.sum().mean()

# 卖出去多少种条目
chipo['item_name'].value_counts().count()

# occupation
users = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|',
                      index_col='user_id')
users2 = pd.read_table('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/u.user.txt')

# 看前25行数据
users.head(25)

# 看最后10行数据
users.tail(5)

# 有多少条数据,
users.shape[0]
# 有多少行数据
users.shape[1]

# 显示所有列名
users.columns

# 数据是如何索引的
users.index

# 每列数据的属性
users.dtypes
users.info

# 显示occupation
users['occupation']

# 有多少种职业
users['occupation'].value_counts().count()
users['occupation'].nunique()

# 比例最高的职业
users['occupation'].value_counts(normalize=True, ascending=False)
# 出现次数最多的职业
users['occupation'].value_counts().sort_values(ascending=False)

# 数据概述
'''Notice: By default, only the numeric columns are returned.'''
users.describe()

# 所有列的数据概述
users.describe(include = 'all')

# 只对occupation做概述
users['occupation'].describe()

# 使用者的平均年龄
round(users['age'].mean(), 2)

# 使用最少的年龄是
users['age'].value_counts(ascending=True)

# World Food Facts
import pandas as pd
import numpy as np

# get data
food = pd.read_csv('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/en.openfoodfacts.org.products.tsv', sep='\t')
# 前五行
food.head(5)
# 有多少观测值
food.shape[0]

# 有多少列
food.shape[1]
food.info()

# 列名
food.columns

# 第105列的列名
food.columns[104]

# 第105列值的类型
food[food.columns[104]]

# 索引方式
food.index

# 第十九个观测值的物品名称
food['product_name'][18]
