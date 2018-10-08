# MPG Cars
import pandas as pd
import numpy as np

# 读取数据
cars1 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv')
cars2 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv')

# 去掉cars1中的空值
cars1 = cars1.loc[:, 'mpg': 'car']

# 每个数据集的观测值
cars1.shape
cars2.shape

# 合并car1 和 car2
cars = cars1.append(cars2)

# 缺失了owner这一列，建立一个随机series 15000到73000
nr_owners = np.random.randint(15000, 73000, size = 398, dtype = 'l')

# 将owners列放入cars
cars['owners'] = nr_owners

# Fictitous Names
import pandas as pd
import numpy as np

# 创建数据
raw_data_1 = {'subject_id': ['1', '2', '3', '4', '5'],
              'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
              'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
raw_data_2 = {'subject_id': ['4', '5', '6', '7', '8'],
              'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
              'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
raw_data_3 = {'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
              'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}

# 创建数据框
data1 = pd.DataFrame(raw_data_1)
data2 = pd.DataFrame(raw_data_2)
data3 = pd.DataFrame(raw_data_3)

# 把两个数据框合并成一个数据框，按行
all_data = pd.concat([data1, data2])

# 按列合并
all_data_col = pd.concat([data1, data2], axis = 1)

# 输出data3
data3

# 合并all_data 和 data3，按照subject_id
pd.merge(all_data, data3, on = 'subject_id')

# 合并data1 和 data2 按照subject_id, 只要有相同id的数据
pd.merge(data1, data2, on = 'subject_id', how = 'inner')

# 合并data1 和 data2的所有数据，按照 subject_id
pd.merge(data1, data2, on = 'subject_id', how = 'outer')

# Housing Market
import pandas as pd
import numpy as np

# 创建3个长度为100的series
s1 = pd.Series(np.random.randint(1, 4, size=100, dtype='l'))
s2 = pd.Series(np.random.randint(1, 3, size=100, dtype='l'))
s3 = pd.Series(np.random.randint(10000, 30000, size=100, dtype='l'))

# 按照列的形式合并3个series
housemkt = pd.concat([s1, s2, s3], axis=1)

# 将列名修改为 bedrs， bathrs， price_sqr_meter
housemkt.rename(columns={0: 'bedrs', 1: 'bathrs', 2: 'price_sqr_meter'})

# 创建一个包含3个series的一列数据框， 命名为bigcolumn
bigcolumn = pd.concat([s1, s2, s3])
bigcolumn = pd.DataFrame(bigcolumn)

# 求长度
len(bigcolumn)

# 重新索引
bigcolumn.reset_index(drop=True, inplace=True)