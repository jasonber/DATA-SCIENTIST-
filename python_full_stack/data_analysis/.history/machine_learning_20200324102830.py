import pandas as pd 
import numpy as np 

dataset = pd.read_csv('./data/ML/1Data_Preprocessing/Data.csv')
ds2 = dataset.copy()
ds2.dropna(inplace=True)
ds2.shape
x = dataset.iloc[:, :-1]

from sklearn.preprocessing import Imputer

