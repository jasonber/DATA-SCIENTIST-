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

[pandas.DataFrame.sort_values](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html)

```python
df = pd.DataFrame({
 'col1' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2' : [2, 1, 9, 8, 7, 4],
   'col3': [0, 1, 9, 4, 2, 3],})
print(df)
df.sort_values(by=['col1'])

```

使用场景：

1、sort sorted 均为python的内置函数

sort 只应用list

sorted应用与所有可迭代的对象

[sorted的参数reverse](http://www.runoob.com/python/python-func-sorted.html)

[sorted sort reverse reversed](https://www.jb51.net/article/78451.htm)

reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。用什么样的值进行排列

2、sort_values 是pandas的函数 只应用与dataframe

# [pandas.select_dtypes()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html)

```python
DataFrame.select_dtypes(include=None, exclude=None)
```

type() 查看数据结构

dytpe查看数据类型 适用于series

dtyps查看数据类型 适用于dataframe

# iloc loc ix

[前闭后开](https://blog.csdn.net/slvher/article/details/44703185)：前闭后开区间（即begin <= idx < end）

loc——通过行标签索引行数据
iloc——通过行号索引行数据，只能是数字

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

总结一下:
data = data['盈利'].copy()得到的data是series,对应的sort_values方法只需要指定axis;
data = data[['盈利']].copy()得到的data是dataframe,对应的sort_values方法只需要指定cols

对列操作

```python
chipo['itme_price'].sum()
# 结果相同，等价，均为series
chipo.item_price.sum()
```



# [pandas.DataFrame.applymap](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.applymap.html)

```python
DataFrame.applymap(func)
```

在数据框的每个元素上使用func

# [apply](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html)

函数格式为：apply(func,*args,**kwargs)
用途：当一个函数的参数存在于一个元组或者一个字典中时，用来间接的调用这个函数，并肩元组或者字典中的参数按照顺序传递给参数。应用于数据框的一行或一列。

# [map](http://www.runoob.com/python/python-func-map.html)

只有series能调用map的api， python的内置函数

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

https://blog.csdn.net/maymay_/article/details/80253068

DataFrame.align(other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, limit=None, fill_axis=0, broadcast_axis=None)

需要仔细研究

```python
>>> data1 = pd.DataFrame(np.ones((6, 3), dtype = float), columns = ['a', 'b', 'c'], index = pd.date_range('6/12/2012', periods = 6))
>>> data2
              d    e    f
2012-06-12  2.0  2.0  2.0
2012-06-13  2.0  2.0  2.0
2012-06-14  2.0  2.0  2.0
2012-06-15  2.0  2.0  2.0
2012-06-16  2.0  2.0  2.0
2012-06-17  2.0  2.0  2.0


>>> data2 = pd.DataFrame(np.ones((6, 3), dtype = float) * 2, columns = ['d', 'e', 'f'], index = pd.date_range('6/12/2012', periods = 6))
>>> data2
              d    e    f
2012-06-12  2.0  2.0  2.0
2012-06-13  2.0  2.0  2.0
2012-06-14  2.0  2.0  2.0
2012-06-15  2.0  2.0  2.0
2012-06-16  2.0  2.0  2.0
2012-06-17  2.0  2.0  2.0

>>> data1.align(data2, join = "outer")
(              a    b    c   d   e   f
2012-06-12  1.0  1.0  1.0 NaN NaN NaN
2012-06-13  1.0  1.0  1.0 NaN NaN NaN
2012-06-14  1.0  1.0  1.0 NaN NaN NaN
2012-06-15  1.0  1.0  1.0 NaN NaN NaN
2012-06-16  1.0  1.0  1.0 NaN NaN NaN
2012-06-17  1.0  1.0  1.0 NaN NaN NaN,              a   b   c    d    e    f
2012-06-12 NaN NaN NaN  2.0  2.0  2.0
2012-06-13 NaN NaN NaN  2.0  2.0  2.0
2012-06-14 NaN NaN NaN  2.0  2.0  2.0
2012-06-15 NaN NaN NaN  2.0  2.0  2.0
2012-06-16 NaN NaN NaN  2.0  2.0  2.0
2012-06-17 NaN NaN NaN  2.0  2.0  2.0)

>>> data1.align(data2, join = "inner")
(Empty DataFrame
Columns: []
Index: [2012-06-12 00:00:00, 2012-06-13 00:00:00, 2012-06-14 00:00:00, 2012-06-15 00:00:00, 2012-06-16 00:00:00, 2012-06-17 00:00:00], Empty DataFrame
Columns: []
Index: [2012-06-12 00:00:00, 2012-06-13 00:00:00, 2012-06-14 00:00:00, 2012-06-15 00:00:00, 2012-06-16 00:00:00, 2012-06-17 00:00:00])

>>> data1.align(data2, join = "left")
(              a    b    c
2012-06-12  1.0  1.0  1.0
2012-06-13  1.0  1.0  1.0
2012-06-14  1.0  1.0  1.0
2012-06-15  1.0  1.0  1.0
2012-06-16  1.0  1.0  1.0
2012-06-17  1.0  1.0  1.0,              a   b   c
2012-06-12 NaN NaN NaN
2012-06-13 NaN NaN NaN
2012-06-14 NaN NaN NaN
2012-06-15 NaN NaN NaN
2012-06-16 NaN NaN NaN
2012-06-17 NaN NaN NaN)

>>> data1.align(data2, join = "right")
(             d   e   f
2012-06-12 NaN NaN NaN
2012-06-13 NaN NaN NaN
2012-06-14 NaN NaN NaN
2012-06-15 NaN NaN NaN
2012-06-16 NaN NaN NaN
2012-06-17 NaN NaN NaN,               d    e    f
2012-06-12  2.0  2.0  2.0
2012-06-13  2.0  2.0  2.0
2012-06-14  2.0  2.0  2.0
2012-06-15  2.0  2.0  2.0
2012-06-16  2.0  2.0  2.0
2012-06-17  2.0  2.0  2.0)

# 返回结果会自动将结果分为两个表。两个表的行是一样的。起到了对齐行名的功能
# axis 1 列
# 因为列名不同，所以无交集，所以empty
>>> left, right = data1.align(data2, join = "inner", axis = 1)
>>> left
Empty DataFrame
Columns: []
Index: [2012-06-12 00:00:00, 2012-06-13 00:00:00, 2012-06-14 00:00:00, 2012-06-15 00:00:00, 2012-06-16 00:00:00, 2012-06-17 00:00:00]
>>> right
Empty DataFrame
Columns: []
Index: [2012-06-12 00:00:00, 2012-06-13 00:00:00, 2012-06-14 00:00:00, 2012-06-15 00:00:00, 2012-06-16 00:00:00, 2012-06-17 00:00:00]


>>> left, right = data1.align(data2, join = "inner", axis = 0)
>>> left
              a    b    c
2012-06-12  1.0  1.0  1.0
2012-06-13  1.0  1.0  1.0
2012-06-14  1.0  1.0  1.0
2012-06-15  1.0  1.0  1.0
2012-06-16  1.0  1.0  1.0
2012-06-17  1.0  1.0  1.0
>>> right
              d    e    f
2012-06-12  2.0  2.0  2.0
2012-06-13  2.0  2.0  2.0
2012-06-14  2.0  2.0  2.0
2012-06-15  2.0  2.0  2.0
2012-06-16  2.0  2.0  2.0
2012-06-17  2.0  2.0  2.0

>>> data1.align(data2, join = "inner", axis = 0)
(              a    b    c
2012-06-12  1.0  1.0  1.0
2012-06-13  1.0  1.0  1.0
2012-06-14  1.0  1.0  1.0
2012-06-15  1.0  1.0  1.0
2012-06-16  1.0  1.0  1.0
2012-06-17  1.0  1.0  1.0,               d    e    f
2012-06-12  2.0  2.0  2.0
2012-06-13  2.0  2.0  2.0
2012-06-14  2.0  2.0  2.0
2012-06-15  2.0  2.0  2.0
2012-06-16  2.0  2.0  2.0
2012-06-17  2.0  2.0  2.0)

```



# pandas levels

levels表示dataframe中多重索引的层次

# [plt.style](http://nbviewer.jupyter.org/github/lijin-THU/notes-python/blob/master/06-matplotlib/06.02-customizing-plots-with-style-sheets.ipynb)

作图时使用什么风格

```python
plt.style.available # 查询哪些风格可用
```



# [KDE](http://www.dataivy.cn/blog/%E6%A0%B8%E5%AF%86%E5%BA%A6%E4%BC%B0%E8%AE%A1kernel-density-estimation_kde/)

kernel density estimation plot 核密度估计

http://seaborn.pydata.org/generated/seaborn.kdeplot.html

非参数估计

由于核密度估计方法不利用有关数据分布的先验知识，对数据分布不附加任何假定，是一种从数据样本本身出发研究数据分布特征的方法，因而，在统计学理论和应用领域均受到高度的重视。

**如何解释图形**

一个变量在另一个变量下的分布情况

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

# [pd.merge](https://blog.csdn.net/starter_____/article/details/79198137)

与 SQL join的作用和用法相同

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html

```python
DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
```

**how** : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’

> - left: use only keys from left frame, similar to a SQL left outer join; preserve key order
> - right: use only keys from right frame, similar to a SQL right outer join; preserve key order
> - outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically **并集**
> - inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys **交集**

[merge, join, concat](https://pandas.pydata.org/pandas-docs/stable/merging.html)

concat 中文https://blog.csdn.net/stevenkwong/article/details/52528616

merge， join中文 https://blog.csdn.net/stevenkwong/article/details/52540605#comments

merge()：与Sql中的join功能一样，关键参数 数据集df，方式how，关键值on

concat()：不同的轴做简单融合，不去重。与append功能相同，是pandas中的方法

append()：是series和dataframe的方法。两个表合并在一起，按照相同的列排在下面，索引不去重。只是简单的将一个表的数据放在另一个表下面。

join()：dataframe内置的join方法是一种快速合并的方法。它默认以index作为对齐的列。



如何区分使用场景？https://zhuanlan.zhihu.com/p/38184619：

pandas.concat——可沿一条轴将多个对象链接到一起；

pandas.merge——可根据一个或多个键将不同的DataFrame中的行连接起来。

append——将dataframe附在数据下面，左面。横向和纵向同时扩充，不考虑columns和index

join——如果为’inner’得到的是两表的交集，如果是outer，得到的是两表的并集；如果有join_axes的参数传入，可以指定根据那个轴来对齐数据。

combine—first可以将重复数据编排在一起，用一个对象中的值填充另一个对象中的值。

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

# [lambda](https://blog.csdn.net/zjuxsl/article/details/79437563) 

 lambda argument_list: expression

```python
new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
```

# [Light Gradient Boosting Machine](http://lightgbm.apachecn.org/cn/latest/)

https://www.jianshu.com/p/b4ac0596e5ef 

LGBM sklearn api

```python
lightgbm.LGBMModel(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=10, max_bin=255, subsample_for_bin=200000, objective=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, **kwargs)
```

```python
fit(X, y, sample_weight=None, init_score=None, group=None, eval_set=None, eval_names=None, eval_sample_weight=None, eval_init_score=None, eval_group=None, eval_metric=None, early_stopping_rounds=None, verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None)
```

verbose: 输出训练内容，在开始实验的时候一般看看训练内容更容易帮助找到改进方法。

verbose_eval输出评估信息，如果设置为True输出评估信息，设置为数字，如5则每5次评估输出一次。



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

# [XGBOOST](https://xgboost.readthedocs.io/en/latest/index.html)

```python
 class xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, **kwargs)
```

```python
 fit(X, y, sample_weight=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None)
```

## 原生LGB、XGB 与 sklearn 接口的关系

https://blog.csdn.net/PIPIXIU/article/details/80463565   

原生版本更灵活，而sklearn版本能够使用sklearn的Gridsearch

对比预测结果，原生xgb与sklearn接口的训练过程相同，结果也相同。   不同之处在于：     

1. 原生采用`xgb.train()`训练，sklearn接口采用`model.fit()`  。     

2. sklearn接口中的参数n_estimators在原生xgb中定义在`xgb.train()`的`num_boost_round`     

3. sklearn`watchlist`为`[(xtrain,ytrain),(xtest,ytest)]`形式，而原生则是`ain,'train'),(dtest,'test')]`,在数据和标签都在DMatrix中，元组里可以定位输出时的名字
各有利弊 如果是为了快速得到模型还是用sklearn API，如果是为了模型更好就用原生

# [hyperopt调参](https://www.e-learn.cn/content/python/736527)

TypeError: 'generator' object is not subscriptable

[Hyperopt TypeError: 'generator' object is not subscriptable](https://blog.csdn.net/FontThrone/article/details/79012616)

[SVC hyperopt调参](https://blog.csdn.net/gg_18826075157/article/details/78068086)

[XGB hyperopt调参](https://www.cnblogs.com/gczr/p/7156270.html)  
[Hyperopt sklearn API](https://github.com/hyperopt/hyperopt-sklearn)  
[参数说明](https://github.com/hyperopt/hyperopt/wiki)

# 其他调参



# [reset_index](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html)

 可以看到，原来的一列index现在变成了columns之一，新的index为[0,1,2,3,4,5]
如果添加参数 reset_index(drop=True) 那么原index会被丢弃，不会显示为一个新列。

```python
result2 = result.reset_index(drop=True)
```

http://www.30daydo.com/article/257

https://blog.csdn.net/jingyi130705008/article/details/78162758

# [pandas.DataFrame.set_index](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.set_index.html)

```python
DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
```

drop 真假的作用？是否保留原来的index



# [pandas.concat](https://blog.csdn.net/stevenkwong/article/details/52528616)

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html

参数说明 
objs: series，dataframe或者是panel构成的序列lsit 
axis： 需要合并链接的轴，0是行，1是列 
join：连接的方式 inner，或者outer

其他一些参数不常用，用的时候再补上说明。

# [any 和 all](https://blog.csdn.net/cython22/article/details/78829288)

本质上讲，any()实现了或(OR)运算，而all()实现了与(AND)运算。
对于any(iterables)，如果可迭代对象iterables（至于什么是可迭代对象，可关注我的下篇文章）中任意存在每一个元素为True则返回True。特例：若可迭代对象为空，比如空列表[]，则返回False。 

对于all(iterables)，如果可迭代对象iterables中所有元素都为True则返回True。特例：若可迭代对象为空，比如空列表[]，则返回True。 

# [value_counts(), count, size](https://blog.csdn.net/qq_20412595/article/details/79921849)

pandas.Series.value_counts

```python
Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
```

Returns object containing counts of unique values.

count pandas

pandas.DataFrame.count

```python
pandas.DataFrame.count(*axis=0*, *level=None*, *numeric_only=False*)
```

Count non-NA cells for each column or row.

[pandas.Series.count](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.count.html)

Return number of non-NA/null observations in the Series

[count python](http://www.runoob.com/python/att-string-count.html)

```python
str.count(sub, start= 0,end=len(string))
```

[size numpy](https://blog.csdn.net/qq_25436597/article/details/79079435)

# [pandas.DataFrame.size](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.size.html)

DataFrame.size
Return an int representing the number of elements in this object.

Return the number of rows if Series. Otherwise return the number of rows times number of columns if DataFrame.

返回数据框中所有的数据个数

```python
>>> s = pd.Series({'a': 1, 'b': 2, 'c': 3})
>>> s.size
3

>>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
>>> df.size
4
```



# [pandas.Series.str.split](https://pandas.pydata.org/pandas-docs/version/0.23.3/generated/pandas.Series.str.split.html)

```python
Series.str.split(pat=None, n=-1, expand=False)
```

# [pandas.to_numeric](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html)

```python
pandas.to_numeric(arg, errors='raise', downcast=None)
```

pandas.DataFrame.astype 感觉这个好用

# [pandas.DataFrame.describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)

```python
DataFrame.describe(percentiles=None, include=None, exclude=None)
```

Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding `NaN` values.



# [pandas.DataFrame.values](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.values.html)

返回一个n维数组，先行后列



# [str.startswith](https://docs.python.org/3/library/stdtypes.html?highlight=startswith#str.startswith)

```python
str.startswith(prefix[, start[, end]])
```

python的内置函数，是str类型数据的方法之一。



# [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)

要注意列名、索引的顺序

# set_index

设定dataframe的索引名字

# [逻辑运算与位运算](http://www.runoob.com/python3/python3-basic-operators.html#ysf4)

or and是逻辑运算符， 均返回的为真的表达式，而不是 1,0

| & 是位运算符，是按照二进制各个位置的数来对比的。



# [*pandas.DataFrame.unstack*](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html)

https://www.cnblogs.com/bambipai/p/7658311.html

不会也不懂。。。。。

# [upper](http://www.runoob.com/python/att-string-upper.html)

# [is 和 ==的区别](http://www.iplaypy.com/jinjie/is.html)

==对比的是值

is 对比的是内存地址

id() 函数用于获取对象的内存地址。



# [pandas.to_datetime](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html)

```python
pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=False)
```

# [pandas.DataFrame.resample](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)

```python
DataFrame.resample(rule, how=None, axis=0, fill_method=None, closed=None, label=None, convention='start', kind=None, loffset=None, limit=None, base=0, on=None, level=None)
```

对时间索引进行频次转换

```python
'''Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.'''
# resample data to 'W' week and use the functions
weekly = data.resample('W').agg(['min','max','mean','std'])

# slice it for the first 52 weeks and locations
weekly.loc[weekly.index[1:53], "RPT":"MAL"] .head(10)
```



[offset—aliases](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

A number of string aliases are given to useful common time series frequencies. We will refer to these aliases as offset aliases.

AS 年份开始的频次， 10AS表示每10年分一组

# [pandas.DataFrame.idxmax](https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.idxmax.html)

```python
DataFrame.idxmax(axis=0, skipna=True)
```

idxmax在dataframe中返回的是一个series， 在series中返回的是个列的标签

idxmax返回的最大值中第一次出现的label

skipna = False 返回的是nan空值

# [np.random](https://www.jianshu.com/p/214798dd8f93)

[【Python】区分python中random模块的randint与numpy.random模块的randint](https://blog.csdn.net/ztf312/article/details/77871424)

random.randint(a, b)     # 返回闭区间 [a, b] 范围内的整数值

numpy.random.randint(a, b)   # 返回开区间 [a, b) 范围内的整数值

# [pandas.DataFrame.rename](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html)

```python
DataFrame.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None)
```

可以修改索引，修改列名

# [pandas.Series.to_frame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.to_frame.html)

# [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)

```python
DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
```

# panda 描述统计函数

https://blog.csdn.net/claroja/article/details/65445063

https://blog.csdn.net/pipisorry/article/details/25625799

axis的选择

0：列固定，按照索引的顺序操作

1：行固定，按照列的顺序操作

# datetime

https://docs.python.org/2/library/datetime.html

内容不少 用到再看吧

https://blog.csdn.net/weixin_38168620/article/details/79596564

datetime的数据类型，可以直接调用年月日

```python
data.loc[data.index.month == 1].mean()
```



# [pandas.Index.flatten](http://pandas.pydata.org/pandas-docs/version/0.14/generated/pandas.Index.flatten.html)

将多维数据按照要求，拉平形成一维数组。与numpy.ndarry.flatten()作用相同

```python
Index.flatten(order='C')
```

参数order : {‘C’, ‘F’, ‘A’}, optional

 C (row-major), The default is ‘C’.

Fortran (column-major) order, or preserve the C/Fortran ordering from a. 

A是什么？ ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.

# PeriodIndex 要搞明白?

[offset—aliases](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

A number of string aliases are given to useful common time series frequencies. We will refer to these aliases as offset aliases.

```python
# 找到每个地方一月份的平均风速
data.loc[data.index.month == 1].mean()

# 以年为频次统计每个地区
# periodIndex 需要学习
data.groupby(data.index.to_period('A')).mean()

# 以月为频次统计每个地方的平均风速
data.groupby(data.index.to_period('M')).mean()

# 以周为频次统计每个地方的平均风速
data.groupby(data.index.to_period('W')).mean()
```



# [python.collections](https://docs.python.org/2/library/collections.html)

具体讲解

https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001411031239400f7181f65f33a4623bc42276a605debf6000

# [pandas.DataFrame.from_dict](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.from_dict.html)

从字典转换为数据框

orient 表示dic 里的key 作为索引名，还是列名。

columns可以直接设置列名

*Parameters:* data, orient, dtype, columns 

Return: pandas.DataFrame

# [pandas.DataFrame.plot](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html)

使用matplot和pylab来作图

```python
DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False, sharex=None, sharey=False, layout=None, figsize=None, use_index=True, title=None, grid=None, legend=True, style=None, logx=False, logy=False, loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False, **kwds)
```

参数：kind : str

‘line’ : line plot (default)
‘bar’ : vertical bar plot
‘barh’ : horizontal bar plot
‘hist’ : histogram
‘box’ : boxplot
‘kde’ : Kernel Density Estimation plot
‘density’ : same as ‘kde’
‘area’ : area plot
‘pie’ : pie plot
‘scatter’ : scatter plot
‘hexbin’ : hexbin plot

# [matplotlib.pyplot.scatter](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html)

```python
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)
```

s 代表marker的大小 其他的作图函数的参数也是一样的

