# alcohol_consumption
import pandas as pd

# get dataset
drinks = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv', sep =',')

# 哪个大洲的啤酒平均饮用较大
drinks.groupby('continent')['beer_servings'].mean().sort_values(ascending=False)

# 统计每个大洲的红酒消费
drinks.groupby('continent')['wine_servings'].describe()

# 每个大洲平均的酒精消费（所有列的数据）
drinks.groupby('continent').mean()

# 中位数
drinks.groupby('continent').median()

# 各大洲spirit 的平均数 最小 最大
drinks.groupby('continent')['spirit_servings'].agg(['mean', 'max', 'min'])

# occupation
import pandas as pd
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep = '|',
                    index_col = 'user_id')

# 每个职位的平均年龄
users.groupby('occupation')['age'].mean()

# 每个职位的男性比例 并从最大到最小排序
gender_total = users.groupby('occupation')['gender'].count()
M_frequency = users[users['gender'] == 'M'].groupby('occupation')['gender'].count()
M_ratio = M_frequency / gender_total * 100
M_ratio.sort_values(ascending= False)

''' method 2
# create a function
def gender_to_numeric(x):
    if x == 'M':
        return 1
    if x == 'F':
        return 0

# apply the function to the gender column and create a new column
users['gender_n'] = users['gender'].apply(gender_to_numeric)


a = users.groupby('occupation').gender_n.sum() / users.occupation.value_counts() * 100 

# sort to the most male 
a.sort_values(ascending = False)'''

# 计算每个职业的最小年龄和最大年龄
users.groupby('occupation')['age'].agg(['min', 'max'])

# 计算每个职业的性别中的，平均年龄
users.groupby(['occupation', 'gender'])['age'].mean()

# 计算每个职业的男女比例
# F_frequency = users[users['gender'] == 'F'].groupby('occupation')['gender'].count()
occupation_gender = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
gender_total2 = users.groupby('occupation')['gender'].agg({'gender':'count'})
occupation_gender_ratio = occupation_gender / gender_total2 * 100

''' method 2
# create a data frame and apply count to gender
gender_ocup = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})

# create a DataFrame and apply count for each occupation
occup_count = users.groupby(['occupation']).agg('count')

# divide the gender_ocup per the occup_count and multiply per 100
occup_gender = gender_ocup.div(occup_count, level = "occupation") * 100

# present all rows from the 'gender column'
occup_gender.loc[: , 'gender']
'''
