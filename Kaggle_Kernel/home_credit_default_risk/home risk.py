import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
# suppress warings
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("/home/hungery/Documents/Kaggle_competion/home_credit/"))

app_train = pd.read_csv("/home/hungery/Documents/Kaggle_competion/home_credit/application_train.csv")