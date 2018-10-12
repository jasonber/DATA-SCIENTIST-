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

# 排除负的quatity条目
online_rt_ex_ne = online_rt[online_rt['Quantity'] > 0]

# 创建一个散点图。前3的国家，每个customer_id的平均UnitPrice的quantity前三的国家
customers = online_rt.groupby(['CustomerID', 'Country']).sum()
customers = customers[customers['UnitPrice'] > 0]
customers['Country'] = customers.index.get_level_values(1)
top_countries = ['Netherlands', 'EIRE', 'Germany']
customers = customers[customers['Country'].isin(top_countries)]
graph = sns.FacetGrid(customers, col='Country')
graph.map(plt.scatter, "Quantity", "UnitPrice", alpha=1)
graph.add_legend()

# Scores
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
raw_data = {'first_name': ['Jason', 'Molly', 'Tian', 'Jake', 'Amy'],
            'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
            'female': [0, 1, 1, 0, 1],
            'age': [42, 52, 36, 24, 73],
            'preTestScore': [4, 24, 31, 2, 3],
            'postTestScore': [25, 94, 57, 62, 70]}

df = pd.DataFrame(raw_data, columns=['first_name', 'last_name', 'female', 'age', 'preTestScore', 'postTestScore'])

# 创建preTestScore 和 postTestScore的散点图， 每个点的大小由age决定
plt.scatter(df['preTestScore'], df['postTestScore'], s=df.age)
plt.title('preTestScore x postTestScore')
plt.xlabel('preTestScore')
plt.ylabel('postTestScore')

# 创建preTestScore 和 postTestScore的散点图 大小是性别的4.5倍 由性别决定颜色
plt.scatter(df['preTestScore'], df['postTestScore'], s=4.5*df['postTestScore'], c=df['female'])
plt.title('preTestScore X postTestScore')
plt.xlabel('preTestScore')
plt.ylabel('postTestScore')

# Tips
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
tips = pd.read_csv('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/07_Visualization/Tips/tips.csv')

# 删除Unnamed: 0列
del tips['Unnamed: 0']

# 画total_bill的直方图
plt.hist(tips['total_bill'], bins=10)
ttbill = sns.distplot(tips['total_bill'], kde=True)
ttbill.set(xlabel='Value', ylabel='Frequency', title='Total Bill')
sns.despine()

# 呈现total_bill 和 Tip的相关图
sns.jointplot('total_bill', 'tip', data=tips)

# total_bill tip size三者的关系呈现在一张图上
sns.pairplot(tips)

# days 和 total_bill的相关关系
sns.stripplot('day', 'total_bill', data=tips, jitter=True)

# 用day作为y轴， 用tips作为x轴， 用性别表示不同的点
sns.stripplot('total_bill', 'day', hue='sex', data=tips)

# 箱形图 total_bill 和day，午餐和晚餐
sns.boxplot(x='day', y='total_bill', hue='time', data=tips)

# 创建两个紧邻着的直方图，tip与 晚餐和午餐的。
sns.set(style='ticks')
graph_tip = sns.FacetGrid(tips, col='time')
graph_tip.map(plt.hist, 'tip')

# 创建两个散点图，一个是male 一个是female， total_bill和 tip的关系， 用somker区别
scatter = sns.FacetGrid(tips, col='sex', hue='smoker')
scatter.map(plt.scatter, 'total_bill', 'tip', alpha=.7)
scatter.add_legend()