import pandas as pd
import numpy as np

dataset = pd.read_csv('./data/ML/1Data_Preprocessing/Data.csv')
ds2 = dataset.copy()
ds2.dropna(inplace=True)
ds2.shape

x = dataset.loc[:, :"Salary"].values
y = dataset.loc[:, "Purchased"]

from sklearn import impute
imputer = impute.SimpleImputer(missing_values=np.nan,
                               strategy='mean',
                               verbose=1)  # * 创建映射的实例,采用哪种映射
imputer = imputer.fit(x[:, 1:3])  # * 用数据使映射的实例
x[:, 1:3] = imputer.transform(x[:, 1:3])  # * 根据映射转换数据

