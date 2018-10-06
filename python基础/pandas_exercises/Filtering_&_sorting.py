# chipotle
import pandas as pd
chipo = pd.read_csv('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/chipotle.tsv', sep = '\t')

# 将价格转换为float 并赋值给price
price = [float(value[1: -1]) for value in chipo['item_price']]
# price2 = [lambda x: float(x[1: -1]) for x in chipo['item_price']]

# 将item_price变为数值
chipo['item_price'] = price

# 删除item_name 和 quantity中的重复值
# chipo[['item_name', 'quantity']].drop_duplicates() 只处理指定的两列的重复值
chipo_filtered = chipo.drop_duplicates(['item_name', 'quantity']) # 显示所有数据，并处理指定的两列

# 只选择数量为1的产品
chipo_one_prod = chipo_filtered[chipo_filtered['quantity'] == 1]

# 数量为1的产品中价格在10以上的产品个数
chipo_one_prod[chipo_one_prod['item_price'] > 10].item_name.nunique()

# 每个条目的价格
price_per_item = chipo_one_prod[['item_name', 'item_price']].sort_values('item_price', ascending=False)

# 按照item_name排序
chipo.sort_values(by = 'item_name')

# 最贵的item的数量
chipo.sort_values(by = 'item_price', ascending = False).head(1)

# veggie salad bowl下单的次数
chipo[chipo['item_name'] == 'Veggie Salad Bowl']['item_name'].count()

# 有多少人不止一次点了 Canned soda
chipo[(chipo['item_name'] == 'Canned Soda') & (chipo['quantity'] > 1)]['order_id'].count()

# Euro 12
import pandas as pd
euro12 = pd.read_csv('https://raw.githubusercontent.com/jokecamp/FootballData/master/UEFA_European_Championship/Euro%202012/Euro%202012%20stats%20TEAM.csv'
                     , sep = ',')

# 只选择 Goal column
euro12['Goals']

# 有多少只队伍参加euro12
euro12['Team'].count()

# 有多少列数据
len(euro12.columns)
euro12.info()

# 只看team， yellow card, red card, 并建立discipline dataframe
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]

# 按照红牌排序 然后是黄牌
discipline.sort_values(by = ['Red Cards', 'Yellow Cards'], ascending = False)

# 每队的平均黄牌数
discipline['Yellow Cards'].mean()

# 进球数在6个以上的队伍
euro12[euro12['Goals'] > 6]

# 以G开头的队伍
euro12[euro12['Team'].str.startswith('G')]

# 选取前7列
euro12.iloc[:, 0: 6]

# 排除最后3行
euro12.iloc[:, 0: -3]

# 选取England, Italy， Russia三国的射门准确率
euro12[euro12.Team.isin(['England', 'Italy', 'Russia'])][['Team', 'Shooting Accuracy']]
euro12.loc[euro12["Team"].isin(["England", "Italy", "Russia"]), ["Team", "Shooting Accuracy"]]

# Fictional Army
import pandas as pd
# 使用字典创建数据
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons',
                         'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
            'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
            'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington',
                       'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

# 创建一个dataframe ，并命名为army
'''按照字典的顺序写入列名，这样才能保证与字典的列索引保持一致。否则pandas会用字母代替列名'''
army2 = pd.DataFrame(raw_data)
army = pd.DataFrame(raw_data, columns=['regiment', 'company', 'deaths', 'battles', 'size', 'veterans', 'readiness',
                                        'armored', 'deserters', 'origin'])

# 使用‘origin'为索引
army = army.set_index('origin')

# 输出veterans列
army['veterans']

# 输出vetrans 和 deaths两列
army[['veterans', 'deaths']]

# 输出所有的列名
army.columns

# 选出 Maine Alaska 中 deaths size deserters数据
army.loc[['Maine', 'Alaska'], ['deaths', 'size', 'deserters']]

# 选择row3到7行，column3到6列的数据
army.iloc[3:7, 3:6]

# 选择第4行以后的数据
army.iloc[3:]

# 选择第4行以前的数据
army.iloc[:3]

# 选择第3列和第7列之间的数据
army.iloc[:, 4: 7]

# 选择deaths大于50的数据
army[army['deaths'] > 50]

# 选择deaths 大于500 或 小于50
'''注意位运算与逻辑运算'''
# army[(army['deaths'] > 500) or (army['deaths'] < 50)] 逻辑运算报错
army[(army['deaths'] > 500) | (army['deaths'] < 50)]

# 选择 Texas 和 Arizona两行的值
army.loc[['Texas', 'Arizona']]

# 选择Arizona行的第三个cell
army.ix[['Arizona'], 2]

# 选择deaths列的第三个cell
army.ix[2, ['deaths']]