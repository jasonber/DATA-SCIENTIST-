# [os](https://www.cnblogs.com/sunyang945/p/7900957.html)

python编程时，经常和文件、目录打交道，这是就离不了os模块。os模块包含普遍的操作系统功能，与具体的平台无关。

 os.listdir()——指定所有目录下所有的文件和目录名。

# [round](http://www.runoob.com/python/func-number-round.html)

round() 方法返回浮点数x的四舍五入值。

```python
print "round(80.23456, 2) : ", round(80.23456, 2)
print "round(100.000056, 3) : ", round(100.000056, 3)
print "round(-100.000056, 3) : ", round(-100.000056, 3)
```

# [sort_values](https://blog.csdn.net/qq_24753293/article/details/80692679)

```python
df = pd.DataFrame({
 'col1' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2' : [2, 1, 9, 8, 7, 4],
   'col3': [0, 1, 9, 4, 2, 3],})
print(df)
df.sort_values(by=['col1'])

```

与sort的区别 没有sort

# iloc loc ix

loc——通过行标签索引行数据 
iloc——通过行号索引行数据

ix——通过行标签或者行号索引行数据（基于loc和iloc 的混合）

```python
print df.loc[:,['c']]

print df.iloc[:,[0]]

print df.ix[:,['c']]

print df.ix[:,[0]]
```

切片

```python
df['name']
df['gender']
df[['name','gender']] #选取多列，多列名字要放在list里
df[0:]	#第0行及之后的行，相当于df的全部数据，注意冒号是必须的
df[:2]	#第2行之前的数据（不含第2行）
df[0:1]	#第0行
df[1:3] #第1行到第2行（不含第3行）
df[-1:] #最后一行
df[-3:-1] #倒数第3行到倒数第1行（不包含最后1行即倒数第1行，这里有点烦躁，因为从前数时从第0行开始，从后数就是-1行开始，毕竟没有-0）

```

```python
# df.loc[index, column_name],选取指定行和列的数据
df.loc[0,'name'] # 'Snow'
df.loc[0:2, ['name','age']] 		 #选取第0行到第2行，name列和age列的数据, 注意这里的行选取是包含下标的。
df.loc[[2,3],['name','age']] 		 #选取指定的第2行和第3行，name和age列的数据
df.loc[df['gender']=='M','name'] 	 #选取gender列是M，name列的数据
df.loc[df['gender']=='M',['name','age']] #选取gender列是M，name和age列的数据

```

```python
df.iloc[0,0]		#第0行第0列的数据，'Snow'
df.iloc[1,2]		#第1行第2列的数据，32
df.iloc[[1,3],0:2]	#第1行和第3行，从第0列到第2列（不包含第2列）的数据
df.iloc[1:3,[1,2]	#第1行到第3行（不包含第3行），第1列和第2列的数据

```



# apply

函数格式为：apply(func,*args,**kwargs)
用途：当一个函数的参数存在于一个元组或者一个字典中时，用来间接的调用这个函数，并肩元组或者字典中的参数按照顺序传递给参数



# [nunique](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.nunique.html)

pandas.DataFrame.nunique

nuinque()是查看该序列(axis=0/1对应着列或行)的不同值的数量。用这个函数可以查看数据有多少个不同值。

**Parameters**:	axis : {0 or ‘index’, 1 or ‘columns’}, default 0 
**dropna** : boolean, default True Don’t include NaN in the counts.

**Returns**: nunique : Series

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
>>> df
   A  B
0  1  1
1  2  1
2  3  1
>>> df.nunique()
A    3
B    1
>>> df.nunique(axis = 1)
0    1
1    2
2    2
dtype: int64
```



# unique

pandas.Series.nunique

只适用于series



# [axis](https://blog.csdn.net/brucewong0516/article/details/79030994)

axis = 0 代表对横轴操作，也就是第0轴；使用0值表示沿着每一列或行标签\索引值向下执行方法
axis = 1 代表对纵轴操作，也就是第1轴；使用1值表示沿着每一行或者列标签模向执行对应的方法

https://blog.csdn.net/weixin_41576911/article/details/79339044

# [format](http://www.runoob.com/python/att-string-format.html)

Python2.6 开始，新增了一种格式化字符串的函数 str.format()，它增强了字符串格式化的功能。

基本语法是通过 {} 和 : 来代替以前的 % 。

```python
>>>"{} {}".format("hello", "world")    # 不设置指定位置，按默认顺序
'hello world'
 
>>> "{0} {1}".format("hello", "world")  # 设置指定位置
'hello world'
 
>>> "{1} {0} {1}".format("hello", "world")  # 设置指定位置
'world hello world'

>>> print("{:.2f}".format(3.1415926))   # 指定格式
3.14
```



# [one hot encoding](https://blog.csdn.net/wl_ss/article/details/78508367)

sklearn OneHOtEncoder 可能无法直接进行onehot编码

get_dummies的优势： 

* 1.本身就是 pandas 的模块，所以对 DataFrame 类型兼容很好. 
* 2.无论你的列是字符型还是数字型都可以进行二值编码. 
* 3.能根据用户指定，自动生成二值编码后的变量名. 
  这么看来，我们找到最完美的解决方案了？ No！get_dummies千般好，万般好，但毕竟不是 sklearn 里的transformer类型，所以得到的结果得手动输入到 sklearn 里的相应模块，也无法像 sklearn 的transformer一样可以输入到pipeline中 进行流程化地机器学习过程。更重要的一点.

注意: get_dummies 不像 sklearn 的 transformer一样，有 transform方法，所以一旦测试集中出现了训练集未曾出现过的特征取值，简单地对测试集、训练集都用 get_dummies 方法将导致数据错误。 



# [align](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.align.html)

pandas.DataFrame.align

需要仔细研究

# [plt.style](http://nbviewer.jupyter.org/github/lijin-THU/notes-python/blob/master/06-matplotlib/06.02-customizing-plots-with-style-sheets.ipynb)

作图时使用什么风格

```python
plt.style.available # 查询哪些风格可用
```



# [KDE](http://www.dataivy.cn/blog/%E6%A0%B8%E5%AF%86%E5%BA%A6%E4%BC%B0%E8%AE%A1kernel-density-estimation_kde/)

kernel density estimation plot 核密度估计

非参数估计

由于核密度估计方法不利用有关数据分布的先验知识，对数据分布不附加任何假定，是一种从数据样本本身出发研究数据分布特征的方法，因而，在统计学理论和应用领域均受到高度的重视。

**如何解释图形**

# [pd.cut](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html)

```python
pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise')
```

# [np.linsapce](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)

创建等差数列

# [plt.xticks](https://blog.csdn.net/henni_719/article/details/77374422)

pyplot的xstick、ystick函数指定坐标轴刻度操作

# [seaborn.heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)

```python
seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
```

# [plt.subplot](https://blog.csdn.net/gt11799/article/details/39103855?locationNum=13)

```python
subplot(nrows, ncols, index, **kwargs)
```

 第一位是行数，第二位是列数，第三位（或者三四位）则是第几个子图。

# [enumerate](http://www.runoob.com/python/python-func-enumerate.html)

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
enumerate(sequence, [start=0])
```

参数
sequence -- 一个序列、迭代器或其他支持迭代对象。
start -- 下标起始位置。

```python
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 小标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

# [plt.tight_layout](https://www.cnblogs.com/nju2014/p/5707980.html)

https://www.jianshu.com/p/91eb0d616adb

tight_layout会自动调整子图参数，使之填充整个图像区域。

# [merge](https://blog.csdn.net/starter_____/article/details/79198137)

与 SQL join的作用和用法相同

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html

```python
DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
```

# [sklearn.preprocessing.MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

```python
min, max = feature_range
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```

归一到 [0, 1]之间

# [sklearn.linear_model.LogisticRegression](https://blog.csdn.net/jark_/article/details/78342644)

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

c是正则项的系数的倒数

# [np.reshape(-1,)](https://blog.csdn.net/wld914674505/article/details/80460042)

数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。

-1表示行/列不知道值，由已知的列/行来计算出

```python
z = np.array([[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]])
z.shape
(4, 4)

z.reshape(-1)
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])

```

# [raise](https://blog.csdn.net/u014148798/article/details/52288326/)

用raise语句来引发一个异常。异常/错误对象必须有一个名字，且它们应是Error或Exception类的子类。

抛出异常，或定义自己认为详细的异常

# [sklearn.KFold()](https://blog.csdn.net/kancy110/article/details/74910185)

shuffle：在每次划分时，是否进行洗牌

若为Falses时，其效果等同于random_state等于整数，每次划分的结果相同

若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的

True每次划分时，都将样本打乱，在划分训练集和验证集

# [pd.shape[0]](http://blog.sina.com.cn/s/blog_4c9dc2a10102vkhd.html)

获取dataFrame的行数和列数，使用的命令是：dataframe.shape[0]和dataframe.shape[1]

# [gc](https://www.cnblogs.com/pinganzi/p/6646742.html)

http://python.jobbole.com/87064/

释放内存

# [Light Gradient Boosting Machine](http://lightgbm.apachecn.org/cn/latest/)

https://www.jianshu.com/p/b4ac0596e5ef 

LGBM sklearn api

```python
lightgbm.LGBMModel(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=10, max_bin=255, subsample_for_bin=200000, objective=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, **kwargs)
```

```python
fit(X, y, sample_weight=None, init_score=None, group=None, eval_set=None, eval_names=None, eval_sample_weight=None, eval_init_score=None, eval_group=None, eval_metric=None, early_stopping_rounds=None, verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None)
```





##针对更快的训练速度

通过设置 bagging_fraction 和 bagging_freq 参数来使用 bagging 方法
通过设置 feature_fraction 参数来使用特征的子抽样
使用较小的 max_bin
使用 save_binary 在未来的学习过程对数据加载进行加速
使用并行学习, 可参考 并行学习指南

##针对更好的准确率

使用较大的 max_bin （学习速度可能变慢）
使用较小的 learning_rate 和较大的 num_iterations
使用较大的 num_leaves （可能导致过拟合）
使用更大的训练数据
尝试 dart

##处理过拟合

使用较小的 max_bin
使用较小的 num_leaves
使用 min_data_in_leaf 和 min_sum_hessian_in_leaf
通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
通过设置 feature_fraction 来使用特征子抽样
使用更大的训练数据
使用 lambda_l1, lambda_l2 和 min_gain_to_split 来使用正则
尝试 max_depth 来避免生成过深的树

# [hyperopt](https://www.e-learn.cn/content/python/736527)

TypeError: 'generator' object is not subscriptable

https://blog.csdn.net/FontThrone/article/details/79012616

# [reset_index](http://www.30daydo.com/article/257)

 可以看到，原来的一列index现在变成了columns之一，新的index为[0,1,2,3,4,5]
如果添加参数 reset_index(drop=True) 那么原index会被丢弃，不会显示为一个新列。

```python
result2 = result.reset_index(drop=True)
```



