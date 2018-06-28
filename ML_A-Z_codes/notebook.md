# part 1 :data preprocessing

**1、概念**

```python
pandas  
numpy  
matplotlib.pyplot  
sklearn.preprocessing.LableEncoder, OneHotEncoder, Imputer, StandardScaler
sklearn.
```

class 类 :  某一类事物 调取的就是类

object 对象 : 某一类事物中的具体实例，带有具体的参数 

method 方法 : 具体实例中的具体动作  通过类的提供的方法来构建一个对象（实例）

dummy variable: 适用于分类变量，方法是onehotencoder ，避免数字带来的大小、等级影响

Menko variable :？

Feature scaling:特征缩放 防止欧式距离带来的误差

##2、操作

1、

```python
import class   

object1 = class(parameter)    

method = object.fit(data1)   

object2 =  method.transform(data2)
```

 2、feature scaling: train and test 均适用train的 度量

## 3、疑问

dummy variable 是否需要scaling ?

是否需要scaling取决你的模型需要多好的解释。虽然scaling有相同的尺度，但是我们会失去dummy data的原始含义。因为dummy data之前只是用来表示分类的，scaling之后就无法表示原来的分类。

dependent variable 是否需要scaling？

按照数据的特点选择是需要scaling。表示分类的不用。其他的需要



# part 2 :regression

##section 4  simple linear regression SLR

**1、概念**

ordinary least square :最小二乘法

**2、操作**

画图

```python
plt.图的形式(x轴，y轴，颜色)
plt.plot(x,y, 颜色)在图上增加其他的内容
plt.title()
plt.xlabel()
plt.ylabel()
plt.show()显示上面的图片 也意味着这次作图结束
```

**3、疑问**

为什么dependent variable 不需要 scaling ？

## section 5 multiple linear regression

statmodels.formula.api

**3、疑问**

为什么X总是不能查看？

因为X中包含了多个类型的变量，它属于object

automatic backward elimination, len(x[0])

2、操作

a、fit的含义

```python
LabelEncoder
OneHotEncoder(categorical_features = 变量列index）
model.fit(data) # fit model(object) to the data
```

b、np.append的特殊用法

```
np.append(arr在哪个数据中插入, values插入什么样的值, axis) 在哪个数据的末尾插入什么样的值，append含义就是在末尾悬挂
```

c、先进行labelencoder 再进行onehotencoder

d、axis = 0 以line加入数据  axis =1 以column加入数据

e、获得模型的统计指标

```
衡量器（object） = sm.衡量方式（）.fit
衡量器.
```

## section 6 polynomial regression

1、不需要 feature scaling

***f、总结***

```
读取数据

预处理
    missing value
         imputer.fit_transform
    labelencoder onehotencoder
    train_test_split
    standardscale
            使用train的量纲
    feature select : BE FS Stepwise
            statmodel 
     model
            model.fit_transform
            model.fit
            moeld.predict
    plt
          plt.图表（x,y ,color）
          plt.plot(x,y ,color)
          plt.tilte()
          plt.xlabel()
          plt.xlabel()
          plt.show()

```

## section 7 SVR

1、框架

先列框架 再coding

```python
# SVR
# Import the libraries
import pandas as pd
import numpy as np

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

# spliting

# feature scaling

# fit SVR to the dataset

# create regressor

# predicting 

# visual the SVR results

import matplotlib.pyplot as plt
plt.scatter(X, y, c = 'green')
plt.show()
```

2、must  feature scaling 

## section 8 Decision Tree Regression

model 需要 x，y同维度

```python
X_grid = np.arange(min(x), max(x), step)
X_grid = X_grid.reshape((len(X_grid),1))
```

## section 9 Random Forest Regression 

.values:Return Series as ndarray or ndarray-like depending on the dtype

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.values.html?highlight=values#pandas.Series.values

不是树越多越平缓，而是在某个树的数目上，RF会收敛

这几中模型是啥？

linear regression

non-linear  regression

non-linear regression non-continuous regression

non-linear or non-continuous regression

## section 10 Evaluating Regression Models Performance

1、$$R^2$$是什么

$$SS_{rest} = \sum{(y_i - \hat{y_i})}^2$$  $$SS_{tot} = \sum{(y_i - y_{avg})}^2$$  $$R^2 = 1 - \frac{SS_{rest}}{SS_{tot}}=\frac{SS_{tot}-SS_{rest}}{SS_{tot}}$$

$$R^2\leq1$$ 越接近1说明预测值与真实值的差距越小，模型拟合的越好。可以为负，模型极其差。所以$$R^2$$ 越大越好

2、调整的$$R^2$$ 是什么

更为全面的指标

多元回归（多个x）。

 $$R^2$$不会下降是固定的，不能很好的指示模型的拟合度

调整$$R^2$$更为敏感，对样本数和X个数均敏感，因为增加了惩罚项



