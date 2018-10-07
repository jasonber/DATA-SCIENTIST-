# Student Alcohol Consumption
import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv')

# 选取‘school’到‘guardian’的数据
stud_alcoh = df.loc[:, 'school': 'guardian']

# 大写字母
capalizer = lambda x: x.upper()

# Mjob 和 Fjob变成大写
stud_alcoh['Mjob'].apply(capalizer)
stud_alcoh['Fjob'].apply(capalizer)

# 显示最后几行值
stud_alcoh.tail()

# 将原表中的Mjob 和 Fjob变成大写
df['Mjob'] = df['Mjob'].apply(capalizer)
df['Fjob'] = df['Fjob'].apply(capalizer)

# 判断非法饮酒， 17岁以上合法, 返回bool值
def majority(age):
    if age > 17:
        return True
    else:
        return False
stud_alcoh['legal_drinker'] = stud_alcoh['age'].apply(majority)

# 数据中的数值扩大10倍
def times10(x):
    if type(x) is int:
        return x*10
    else:
        return x
stud_alcoh.applymap(times10)

# US_Crime_Rates
import pandas as pd
crime = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv")

# 列属性
crime.info()

''' 将year转换为datetime64'''
crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')

# 设置Year为索引
# crime1 = crime.set_index(crime['Year'], drop = True)
# crime2 = crime.set_index(crime['Year'], drop = False)
crime = crime.set_index(crime['Year'])

# 删除Total列
# crime.drop(columns='Total')
#直接删除了源数据中的列
del crime['Total']

# 把年龄按照10年分组，并求和
# 人口列的求和会出有问题，应该用区间内的最大值表示
crimes = crime.resample('10AS').sum()
population = crime['Population'].resample('10AS').max()
crimes['Population'] = population

# 美国最危险的10年
crimes.idxmax(axis = 0)
