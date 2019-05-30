import pandas as pd
from sqlalchemy import create_engine
import os

conn = create_engine("mysql+pymysql://root:123456zz@localhost:3306/review")
path = 'C:/Users/zhang/Documents/my_git/DATA-SCIENTIST-/sql'
root = []
dir = []
file = []

for r, d, f in os.walk(path):
    root.append(r)
    dir.append(d)
    file.append(f)

m = 0
for f in file[0]:
    d = path + '/{}'.format(str(f))
    n = os.path.splitext(file[0][m])
    if n[1] == '.csv':
        print(f)
        df = pd.read_csv(path + '/' + '{}'.format(file[0][m]))
        df.to_sql('{}'.format(n[0]), con=conn, if_exists='replace', index=False)
    m += 1
