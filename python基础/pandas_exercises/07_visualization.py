# chipotle
import pandas as pd
import collections
import matplotlib.pyplot as plt

# 载入数据
chipo = pd.read_csv('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/chipotle.tsv', sep='\t')

# 看前10条数据
chipo.head()

# 购买最多的前五个条目的直方图()
x = chipo['item_name']
letter_counts = collections.Counter(x)
df = pd.DataFrame.from_dict(letter_counts, orient='index', columns=['item_count'])
top_5 = df.sort_values('item_count', ascending=False).head(5)
top_5.plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Item')
plt.title('TOP5 Count Item')

# 做item数量和订单价格的散点图
chipo['item_price'] = [float(value[1:-1]) for value in chipo['item_price']]
orders_no_pr = chipo.groupby('order_id').sum()
# orders_no_pr[].plot(x=['item_price'], y=['quantity'], kind="scatter")
plt.scatter(x=orders_no_pr['item_price'], y=orders_no_pr['quantity'], s=50, c='green')
plt.title('Number of items ordered per order price')
plt.xlabel('order_price')
plt.ylabel('item orderd')

# Online Retails Purchase
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置sns样式
sns.set(style='ticks')
# 载入数据
online_rt = pd.read_csv('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/07_Visualization/Online_Retail/Online_Retail.csv', encoding='cp1252')

# 除了UK外的量最多的top10 国家
country_quantity = online_rt.groupby('Country').sum()
top10_quantity_exUK = country_quantity.drop(index='United Kingdom').sort_values(by='Quantity', ascending=False).head(10)
top10_quantity_exUK['Quantity'].plot(kind='bar')
plt.title('TOP 10 Countries of most quantity')
plt.xlabel('Country')
plt.ylabel('Quantity')