# 一 基础命令
import pandas as pd
import numpy as np
# Create an example dataframe about a fictional army
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
print(raw_data)

# dic tranform to dataframe
army = pd.DataFrame(raw_data)

# set origin as index
army1 = army.set_index(["origin"])

# print the values of veterans
army1['veterans']

# print values of veterans and deaths
army1[["veterans", "deaths"]]

# print the columns
army1.columns

'''
get the values is not Dragoons of regiment
'''
army1.loc[army1["regiment"] != "Dragoons"]

# get values from r3-7 and c3 and c6
army1.iloc[2:6, [2, 6]]

# 二 饮酒习惯
dataframe = pd.read_csv("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/python基础/Student_Alcohol.csv")

# slice school:guardian
dataframe.iloc[:,0:12]

# make the initial of c-Mjob and c-Fjob to be capital
data2 = dataframe.loc[:, ["Mjob", "Fjob"]]
data21 = pd.Series(data2["Mjob"])
data22 = pd.Series(data2["Fjob"])
dataframe["Mjob"] = data21.map(lambda x:x.capitalize())
dataframe["Fjob"] = data22.map(lambda x:x.capitalize())

''' name a column "legal_drinker". Find the drinker who is under 17. They are legal_drinker. '''
# majority = lambda x:1 if x > 17 else 0
majority = lambda x:"合法" if x > 17 else "非法"
dataframe["legal_drinker"] = dataframe["age"].map(majority)

# 二、练习二
users = pd.read_csv("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/python基础/users.csv", sep = "|")

'''get the average of age from every occupation'''
# method 1
Ave_age1 = users.groupby('occupation').age.mean()
# method 2
Ave_age2 = users["age"].groupby(users["occupation"]).mean()
print(Ave_age1, Ave_age2)

'''name the male by 1(male) or 0(female). 
get the percent of male from every occupation named "male_pct" '''
# method 1
def gender_to_numberic(x):
    if x == "M":
        return 1
    if x == "F":
        return 0
users["gender_n"] = users["gender"].apply(gender_to_numberic)

male_pct_occu = users.groupby("occupation").gender_n.sum() / users.occupation.value_counts() * 100
male_pct_occu.sort_values(ascending=False)

# get the min and max of user age
users.groupby("occupation").age.agg(["max", "min"])

import pandas as pd
import numpy as np

chipo = pd.read_csv("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/python基础/chipotle.csv", sep ="\t")

# how many items` price beyond 10 $?
''' hwo to eliminate the $'''
# method 1  use type transform
# prices = [float(value[1: -1]) for value in chipo.item_price]
# method 2 use split
chipo["item_price"] = chipo["item_price"].str.split("$").str[1]
chipo["item_price"] = pd.to_numeric(chipo["item_price"])
(chipo["item_price"] > 10).value_counts()

# how much every project
chipo_filltered = chipo.drop_duplicates(['item_name', 'quantity'])
chipo_one_prod = chipo_filltered[chipo_filltered.quantity == 1]
price_per_item = chipo_one_prod[["item_name", "item_price"]]
price_per_item.sort_values(by = "item_price", ascending = False)

 # Create an example dataframe about a fictional army
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
            'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
            'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

army = pd.DataFrame(raw_data)
army1 = army.set_index(['origin'])

print(army1['veterans'])
print(army1[['veterans', 'deaths']])
print(army1.columns)
army1[army1['regiment'] != 'Dragoons']
army1.iloc[2:7, 2:6]

df = pd.read_csv