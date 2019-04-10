import pandas as pd
from sqlalchemy import create_engine
import re

database = ['interview', 'leetcode' ,'review']
conn = "mysql+pymysql://root:ztzz910327@localhost:3306/"
get_tables = 'show tables;'
table_list = []
for i in database:
    con = create_engine(conn+str(i))
    a = pd.read_sql(get_tables, con=con)
    table_list.append(a)
tables = pd.concat(table_list)
tables.rename(columns={'Tables_in_interview':'interview', 'Tables_in_leetcode':'leetcode',
                       'Tables_in_review':'review'}, inplace=True)

create_command = 'show create table '
command_list = []
for i in database:
    con = create_engine(conn+str(i))
    for j in tables.columns:
        for n in tables[str(i)].dropna():
            b = pd.read_sql(create_command + str(n), con=con)
            command_list.append(b)
command = pd.concat(command_list)
command.drop_duplicates(subset='Create Table', inplace=True)


def format_hive(x):
    pattern = r'COLLATE.*?NULL|ENGINE.*|PRIMARY.*(?=\))|DEFAULT.*?NULL'
    a = re.sub(pattern, '', x)
    return a
command.iloc[:, 1] = command.iloc[:, 1].map(format_hive)
# command.replace(r'COLLATE.*?NULL|ENGINE.*|PRIMARY.*(?=\))|DEFAULT.*?NULL', '', regex=True, inplace=True)
save = '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/'
command.to_csv(save+'create_tables.csv', index=False)


