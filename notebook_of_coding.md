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

切片操作只针对多维矩阵可以使用，list不能使用[:,:]这种方式取数，它也没有这种方式

loc——通过行标签索引行数据
iloc——通过行号索引行数据，只能是数字，[前闭后开](https://blog.csdn.net/slvher/article/details/44703185)：前闭后开区间（即begin <= idx < end）

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

# 切片的复制功能
c = d
d = list(np.arange(0, 10))
c = d
del(c[0])

c
Out[30]: [1, 2, 3, 4, 5, 6, 7, 8, 9]
d
Out[31]: [1, 2, 3, 4, 5, 6, 7, 8, 9]

d = list(np.arange(0, 10))
c = d[:]
c
Out[34]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
d
Out[35]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

del[c[0]]
c
Out[37]: [1, 2, 3, 4, 5, 6, 7, 8, 9]
d
Out[38]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

e = d.copy()
e
Out[40]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
del(e[0])
e
Out[42]: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```python
# df.loc[index, column_name],选取指定行和列的数据
df.loc[0,'name'] # 'Snow' 
df.loc[0:2, ['name','age']] 		 #选取第0行到第2行,共三行，name列和age列的数据, 注意这里的行选取是包含下标的。
# df.loc[0:2]等同于df.iloc[0:3] loc取得是行号，iloc取的是索引
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

```python
总结一下:
data = data['盈利'].copy()得到的data是series,对应的sort_values方法只需要指定axis;
data = data[['盈利']].copy()得到的data是dataframe,对应的sort_values方法只需要指定cols
https://blog.csdn.net/chenKFKevin/article/details/62049060
切片还有复制的作用 
    
type(iris.iloc[10:30, 2:2])
Out[6]: pandas.core.frame.DataFrame
type(iris.iloc[10:30, 2])
Out[7]: pandas.core.series.Series
```

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

nuinque()是查看该序列(axis=0/1对应着列(索引)或行(列))的不同值的数量。用这个函数可以查看数据有多少个不同值。

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

axis = 0 ***索引*** 代表对横轴操作，也就是第0轴；使用0值表示沿着每一列或行标签\索引值向下执行方法
axis = 1 ***列*** 代表对纵轴操作，也就是第1轴；使用1值表示沿着每一行或者列标签模向执行对应的方法

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

# [operator 逻辑运算与位运算](http://www.runoob.com/python3/python3-basic-operators.html#ysf4)

or and是逻辑运算符， 均返回的为真的表达式，而不是 1,0

| & 是位运算符，是按照二进制各个位置的数来对比的。

[operator.itemgetter()](https://blog.csdn.net/dongtingzhizi/article/details/12068205)

itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号）,按照行列的索引提取值

```python
import numpy as np
import operator
list = [[1, 2, 3], 'b', [4, 5, 6]]
array = np.arange(0, 30 , 2)
matrix = array.reshape(5, 3)
a = operator.itemgetter(1)
b = operator.itemgetter(1, 0)
a(matrix)
Out[9]: array([ 6,  8, 10])
b(matrix)
Out[10]: (array([ 6,  8, 10]), array([0, 2, 4]))
c = operator.itemgetter((1, 0), 1)
c(matrix)
Out[13]: (6, array([ 6,  8, 10]))
```

[sorted函数以及operator.itemgetter函数](https://blog.csdn.net/dongtingzhizi/article/details/12068205)

```python
import numpy as np
import operator
a = {}
alp = ['d', 'c', 'b', 'a']
num = range(1, 5)
for i, j in zip(alp, num):
    a[i] = j
a
Out[12]: {'d': 1, 'c': 2, 'b': 3, 'a': 4}

sorted_a_item = sorted(a.items(), key=operator.itemgetter(1), reverse=True)
sorted_a_item
Out[18]: [('a', 4), ('b', 3), ('c', 2), ('d', 1)]

sorted_a_item = sorted(a.items(), key=operator.itemgetter(2), reverse=True)
IndexError: tuple index out of range

sorted_a_item = sorted(a.items(), key=operator.itemgetter(0), reverse=True)
sorted_a_item
Out[21]: [('d', 1), ('c', 2), ('b', 3), ('a', 4)]
sorted_a_item = sorted(a.items(), key=operator.itemgetter(0))
sorted_a_item # tuple组成的list
Out[23]: [('a', 4), ('b', 3), ('c', 2), ('d', 1)]
```



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



# [pandas.Index.get_level_values](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.get_level_values.html)

Return an Index of values for requested level, equal to the length of the index

# [pandas.DataFrame.isin](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isin.html)

Return boolean DataFrame showing whether each element in the DataFrame is contained in values.

# [seaborn.FacetGrid](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)

seaborn 是实现使matplotlib更高级的工具https://seaborn.pydata.org/index.html

**要进一步学习**

This class maps a dataset onto multiple axes arrayed in a grid of rows and columns that correspond to levels of variables in the dataset. The plots it produces are often called “lattice”, “trellis”, or “small-multiple” graphics.

```python
class seaborn.FacetGrid(data, row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=True, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None, size=None)
```

后面必跟着map（[seaborn.FacetGrid.map](https://seaborn.pydata.org/generated/seaborn.FacetGrid.map.html?highlight=map#seaborn.FacetGrid.map)）

```python
FacetGrid.map(func, *args, **kwargs)
```

sns的style： {darkgrid, whitegrid, dark, white, ticks}

# 创建preTestScore 和 postTestScore的散点图 大小是性别的4.5倍 由性别决定颜色

```python
plt.scatter(df['preTestScore'], df['postTestScore'], s=4.5*df['postTestScore'], c=df['female'])
plt.title('preTestScore X postTestScore')
plt.xlabel('preTestScore')
plt.ylabel('postTestScore')
```

原来参数还可以这么用

# [matplotlib.pyplot.hist](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)

```python
matplotlib.pyplot.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)
```

# [seaborn.distplot](https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot)

```python
seaborn.distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)
```

# [seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html?highlight=jointplot#seaborn.jointplot)

```python
seaborn.jointplot(x, y, data=None, kind='scatter', stat_func=None, color=None, height=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)
```

Draw a plot of two variables with bivariate and univariate graphs.

kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional

Kind of plot to draw.

# [seaborn.pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html?highlight=pairplot#seaborn.pairplot)

```python
seaborn.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None, size=None)
```

Plot pairwise relationships in a dataset.

# [seaborn.stripplot](https://seaborn.pydata.org/generated/seaborn.stripplot.html?highlight=stripplot#seaborn.stripplot)

```python
seaborn.stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, jitter=True, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs)
```

Draw a scatterplot where one variable is categorical.

jitter : float, True/1 is special-cased, optional

抖动，方便看清楚数据点的位置，避免因重叠而看不清

hue：可以看出是legend表示的是什么。分类标准



# [matplotlib.pyplot.pie](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html)

```python
matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None, radius=None, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False, *, data=None)
```

shadow：在图下面是否有阴影

explode：分开的距离

startangle：起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看

autopct：对图片里的数值文本格式的设置，同.format()的使用方法一样。

# [matplotlib.pyplot.axis](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axis.html)

设置轴的某些属性：

'on'	Turn on axis lines and labels.
'off'	Turn off axis lines and labels.
'equal'	Set equal scaling (i.e., make circles circular) by changing axis limits.
'scaled'	Set equal scaling (i.e., make circles circular) by changing dimensions of the plot box.
'tight'	Set limits just large enough to show all data.
'auto'	Automatic scaling (fill plot box with data).
'normal'	Same as 'auto'; deprecated.
'image'	'scaled' with axis limits equal to data limits.
'square'	Square plot; similar to 'scaled', but initially forcing xmax-xmin = ymax-ymin.



# [matplotlib.tight_layout](https://matplotlib.org/api/tight_layout_api.html)

tight_layout会自动调整子图参数，使之填充整个图像区域。这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分。

# [seaborn.lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html)

Plot data and regression model fits across a FacetGrid.

This function combines regplot() and FacetGrid. It is intended as a convenient interface to fit regression models across conditional subsets of a dataset.

```python
seaborn.lmplot(x, y, data, hue=None, col=None, row=None, palette=None, col_wrap=None, height=5, aspect=1, markers='o', sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None, legend=True, legend_out=True, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None, size=None)
```

seaborn 比matplot简单很多

还有一个叫做echart的库 这些都要学

# pandas.Series.is_unique

只适用于series，返回true和false 判断是否为唯一。

# [pandas.DataFrame.dropna](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)

```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)[source]
```

# 缺失值统计

````python
df.isnull().sum()
````

# pd.DataFrame.copy()

制作另一份数据

```python
df2 = df # df2是df是视图，修改df也会改变df2
df3 = df.copy() # df3是df备份，修改df不会改变df3
```



# panda + pymysql

```python
import pandas as pd
import pymysql
database = pymysql.connect(host='localhost', user='root', password='ztzz910327', db='competition_data')
sql = 'SELECT * FROM employee'
df = pd.read_sql(sql, database)
database.close()
```



# 如何在python中操作mysql---pymysql

https://blog.csdn.net/qq_35304570/article/details/78767288

https://ask.hellobi.com/blog/wangdawei/9367

```python
#流程
connect
cursor（）
sql语句
execute（）
commit（）
close（）
```



pymysql.Connect()参数说明
host(str):      MySQL服务器地址
port(int):      MySQL服务器端口号
user(str):      用户名
passwd(str):    密码
db(str):        数据库名称
charset(str):   连接编码

connection对象支持的方法
cursor()        使用该连接创建并返回游标
commit()        提交当前事务
rollback()      回滚当前事务
close()         关闭连接

cursor对象支持的方法
execute(op)     执行一个数据库的查询命令
fetchone()      取得结果集的下一行
fetchmany(size) 获取结果集的下几行
fetchall()      获取结果集中的所有行
rowcount()      返回数据条数或影响行数
close()         关闭游标对象

```python
# 示例
import pymysql
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
# SQL 插入语句
sql = "INSERT INTO EMPLOYEE(FIRST_NAME, \
       LAST_NAME, AGE, SEX, INCOME) \
       VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
       ('Mac', 'Mohan', 20, 'M', 2000)
try:
   # 执行sql语句
   cursor.execute(sql)
   # 执行sql语句
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()
# 关闭数据库连接
db.close()

#查询
import pymysql
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
# SQL 查询语句
sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   for row in results:
      fname = row[0]
      lname = row[1]
      age = row[2]
      sex = row[3]
      income = row[4]
       # 打印结果
      print ("fname=%s,lname=%s,age=%d,sex=%s,income=%d" % \
             (fname, lname, age, sex, income ))
except:
   print ("Error: unable to fetch data")
# 关闭数据库连接
db.close()
```





# [TRY](http://www.runoob.com/python/python-exceptions.html)

```python
# 语法
try:
<语句>        #运行别的代码
except <名字>：
<语句>        #如果在try部份引发了'name'异常
except <名字>，<数据>:
<语句>        #如果引发了'name'异常，获得附加的数据
else:
<语句>        #如果没有异常发生

#代码示例
    try:
    fh = open("testfile", "w")
    fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print "Error: 没有找到文件或读取文件失败"
else:
    print "内容写入文件成功"
    fh.close()
# 无异常时的输出
内容写入文件成功
# 异常时的输出
Error: 没有找到文件或读取文件失败

# 还可以不带错误名
try:
    正常的操作
   ......................
except:
    发生异常，执行这块代码
   ......................
else:
    如果没有异常执行这块代码
```

# python 中操作数据库 SQLAlchemy

```python
from SQLAlchemy import *
import pymysql
db = create_engine('mysql+pymysl://用户名：密码@端口/数据库？charset=编码方式', echo=False)
# 提交操作
db.execute('sql语句')
# 展示结果
da.execute('sql语句').fetchall()
```

```python
# 示例 与pandas合作 操作sql
import pandas as pd
from sqlalchemy import *

# 创建SQL连接
database = create_engine('mysql+pymysql://root:密码@localhost/competition_data')
# 读取sql到pandas
sql = 'SELECT * FROM employee'
df = pd.read_sql(sql, database)
# 在df中插入数据
item1 = pd.DataFrame([['xiao', 'zhao', 31, 'M', 0], ['zhang', 'wuji', 27, 'F', 0]])
item1.rename(columns = {0:'first_name', 1:'last_name', 2:'age', 3:'sex', 4:'income'}, inplace=True)
new_df = pd.concat([df, item1], ignore_index=True)
# 将修改好的df 存入sql
new_df.to_sql('employee', con=database, if_exists='append', index=False)
# 查询SQL并展示结果（应该是游标展示吧？）
database.execute("select * from employee").fetchall()
# 在sql中插入一行数据，如果出错就返回sql中的错误
try:
    database.execute("INSERT INTO employee (first_name, last_name, age, sex, income)  \
    VALUE('Chang', 'Wudi', 56, 'F', 3000)")
except Exception as e:
    print(e)
```



## [pandas.DataFrame.to_sql](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html)

```python
DataFrame.to_sql(name, con, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None)
```

# [动态生成变量](https://www.cnblogs.com/dcb3688/p/4347688.html)

https://docs.python.org/3/library/functions.html?highlight=locals#locals

```python
1 createVar = locals()
2 listTemp = range(1,10)
3 for i,s in enumerate(listTemp):
4     createVar['a'+i] = s
5 print a1,a2,a3
6 #......
复制代码
复制代码
1 def foo(args):
2     x=1
3     print locals()
4 
5 foo(123)
6 
7 #将会得到 {'arg':123,'x':1}
复制代码
复制代码
1 for i in range(3):
2     locals()['a'+str(i)]=i
3     print 'a'+str(i)
```

```python
# 按需求的df名字生成df
def get_data(path, data_name):
    create_vars[data_name] = pd.read_csv(path)
    print("Success")
    return create_vars[data_name]


path = "/home/zhangzhiliang/Documents/Kaggle_data/home_risk/"
file_list = ['application_train', 'application_test', 'bureau', 'bureau_balance', 'POS_CASH_balance',
             'credit_card_balance', 'previous_application', 'installments_payments']
df_name = ['app_train', 'app_test', 'bureau', 'bureau_balance', 'cash',
           'credit', 'previous', 'installments']

path_list = []
for i in file_list:
    path_list.append(path + i + '.csv')
# zip 同时遍历两个list
create_vars = locals()
for i, j in zip(path_list, df_name):
    create_vars[j] = get_data(i, j)
```



# [python 内置extend和append](https://blog.csdn.net/kancy110/article/details/77131441)

extend是将object中的各个元素加入到列表中， append是将object作为整体加入到列表中。

```python
key1 = ['SK_ID_CURR']
key2 = key1.copy()
key2.extend(['SK_ID_PREV'])
# extend 和 append不能进行赋值操作,会返回None类型
key2 = extend('SK_ID_PREV')
# 区别
a = [1, 2, 3]
b = [4, 5, 6]
a.append(b) # 输出[1, 2, 3, [4, 5, 6]]
a = [1, 2, 3]
a.extend(b) # 输出[1, 2, 3, 4, 5, 6]

```



# [Featuretools](https://docs.featuretools.com/index.html)

概念：

实体和实体集：实体就是对象+关系，实体集就是把实体装到表里或df中

关系：各个表之间的关联方式，有父表、子表

feature primitives特征基本操作：在一个表或多个表中进行操作创造出新的特征。

1、聚合运算：对每个父表的子表进行统计，均值、最小、最大、标准差等。

2、转换：对一个表中的一列或多列进行操作。

deep feature synthesis深度特征综合：

```python
import featuretools as ft
# 创建实体集
es = ft.EntitySet(id=)
# 加入实体 每个实体都需要有唯一的索引
es = es.entity_from_dataframe(entity_id=, dataframe=, index=)
# 没有索引的 可以由函数自动生成
es = es.entity_from_dataframe(entity_id=, dataframe=, make_index=True, index=)
# 描述各个实体间的关系
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
# 添加关系到实体集
es = es.add_relationship([r_app_bureau])
# 进行
```

注意事项：

1、创建关系时， 每个实体之间的关系最好是单向的，不然特征基本体会堆积。单向、不循环图。



# [python 打开文件的方法](http://www.runoob.com/python/file-methods.html)

http://www.runoob.com/python/python-func-open.html

```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
```

读取txt文件

```python
fr = open('/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/'
          'machine_learning_in_action/3_decision_tree/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_feature = ['age', 'prescript', 'astigmatic', 'tearRate']


def file2matrix(filename): #导入数据
	fr = open(filename)
	arrayOfLines = fr.readlines()
	returnMat = [i.strip().split('\t') for i in arrayOfLines]
	labels = ['age', 'prescript', 'astigmatic', 'tearRate']
	return returnMat,labels
```



# [python 类](http://www.runoob.com/python/python-object.html)

https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/00138682004077376d2d7f8cc8a4e2c9982f92788588322000

类和实例

面向对象最重要的概念就是类（Class）和实例（Instance），必须牢记类是抽象的模板，比如Student类，而实例是根据类创建出来的一个个具体的“对象”，每个对象都拥有相同的方法，但各自的数据可能不同。

类是模板， 描述的是一类事物的样子和功能。样子通过属性定义，功能通过动作来定义。

```python
class ClassName(基类名):
    def __init__(self, attr1， attr2.....):
        self.attr1 =
        self.attr2 =
    def function:
        .....
        return 
    
```

单下划线、双下划线、头尾双下划线说明：

```python
__foo__: 定义的是特殊方法，一般是系统定义名字 ，类似 __init__() 之类的。

_foo: 以单下划线开头的表示的是 protected 类型的变量，即保护类型只能允许其本身与子类进行访问，不能用于 from module import *

__foo: 双下划线的表示的是私有类型(private)的变量, 只能是允许这个类本身进行访问了。
```



# [python pass](https://www.runoob.com/python/python-pass-statement.html)

Python pass是空语句，是为了保持程序结构的完整性。

pass 不做任何事情，一般用做占位语句。

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*- 

# 输出 Python 的每个字母
for letter in 'Python':
   if letter == 'h':
      pass
      print '这是 pass 块'
   print '当前字母 :', letter

print "Good bye!"

# 输出
当前字母 : P
当前字母 : y
当前字母 : t
这是 pass 块
当前字母 : h
当前字母 : o
当前字母 : n
Good bye!
```



# [np.c_ 和 np.r_](https://blog.csdn.net/yj1556492839/article/details/79031693)

np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。

# [sklearn.datasets.make_blobs](https://www.jianshu.com/p/069d8841bd8e)

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

生成聚类数据集的模块

Generate isotropic Gaussian blobs for clustering.

# [np.newaxis](https://blog.csdn.net/lanchunhui/article/details/49725065)

为 numpy.ndarray（多维数组）增加一个轴

# [python operator 模块](https://docs.python.org/2/library/operator.html)

https://blog.csdn.net/hephec/article/details/77992114 

本模块主要包括一些python内部操作符对应的函数，主要包括几类：对象比较，逻辑比较，算术运算和序列操作

# [np.tile](https://blog.csdn.net/wy250229163/article/details/52453201)
https://blog.csdn.net/ksearch/article/details/21388985
函数形式： tile(A，rep) 
功能：重复A的各个维度 
参数类型： 

- A: Array类的都可以 
- rep：A沿着各个维度重复的次数

# [np.argsort](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argsort.html)
https://www.cnblogs.com/yyxf1413/p/6253995.html

返回值排序后的索引，默认升序

# [dic.get()](http://www.runoob.com/python/att-dictionary-get.html)

Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值

```python
dict.get(key, default=None)
```

key -- 字典中要查找的键。
default -- 如果指定键的值不存在时，返回该默认值值。

# [dic.iteritems()](https://blog.csdn.net/program_developer/article/details/78657908)
https://blog.csdn.net/liukai2918/article/details/78307271

字典的items方法作用：是可以将字典中的所有项，以列表方式返回。因为字典是无序的，所以用items方法返回字典的所有项，也是没有顺序的。
字典的iteritems方法作用：与items方法相比作用大致相同，只是它的返回值不是列表，而是一个迭代器。

# [operator.itemgetter()](https://blog.csdn.net/dongtingzhizi/article/details/12068205)
要注意，operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。
```python
a = [1,2,3] 
>>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
>>> b(a) 
2 
>>> b=operator.itemgetter(1,0)   //定义函数b，获取对象的第1个域和第0个的值
>>> b(a) 
(2, 1) 
```

# [np.min()](https://blog.csdn.net/qq_18433441/article/details/54743271)

```python
import numpy as np
a = np.array([[1,5,3],[4,2,6]])
print(a.min()) #无参，所有中的最小值
print(a.min(0)) # axis=0; 每列的最小值
print(a.min(1)) # axis=1；每行的最小值
# 结果
1
[1 2 3]
[1 2]
```



# [字典](http://www.runoob.com/python3/python3-dictionary.html)

# [dataframe.replace](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html)

# [*args 和 **kwargs](https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/)

*args 是用来发送一个非键值对的可变数量的参数列表给一个函数

```python
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')

#out
first normal arg: yasoob
another arg through *argv: python
another arg through *argv: eggs
another arg through *argv: test
```



```python
**kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs。

def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))


>>> greet_me(name="yasoob")
name == yasoob
```

那么如果你想在函数里同时使用所有这三种参数， 顺序是这样的：

```python
some_func(fargs, *args, **kwargs)
```



# python 序列化（持久化）和 sklearn 序列化（持久化）

[python pickle](https://blog.csdn.net/sxingming/article/details/52164249)

pickle提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。

pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，

```python
>>> a1 = 'apple'  
>>> b1 = {1: 'One', 2: 'Two', 3: 'Three'}  
>>> c1 = ['fee', 'fie', 'foe', 'fum']  
>>> f1 = file('temp.pkl', 'wb')  
# 保存
>>> pickle.dump(a1, f1, True)  
>>> pickle.dump(b1, f1, True)  
>>> pickle.dump(c1, f1, True)  
>>> f1.close()
# 载入
>>> f2 = file('temp.pkl', 'rb')  
>>> a2 = pickle.load(f2)  
>>> a2  
'apple'  
>>> b2 = pickle.load(f2)  
>>> b2  
{1: 'One', 2: 'Two', 3: 'Three'}  
>>> c2 = pickle.load(f2)  
>>> c2  
['fee', 'fie', 'foe', 'fum']  
>>> f2.close()  
```

[机器学习模型的持久化 joblib](https://blog.csdn.net/Dream_angel_Z/article/details/47175373)

```python
>>>from sklearn.externals import joblib
>>> os.chdir("workspace/model_save")
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
>>> clf.fit(train_X,train_y)
# 保存模型
>>> joblib.dump(clf, "train_model.m")
# 载入模型
>>> clf = joblib.load("train_model.m")
>>> clf.predit(test_X) #此处test_X为特征集
```



# with as 和 context manage

https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html

http://www.maixj.net/ict/python-with-as-14005

http://python.jobbole.com/81477/

```python
# 使用 with as 不再需要关注 close
try:
    with open( "a.txt" ) as f :
        do something
except xxxError:
    do something about exception
```

# [np.mat](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.mat.html)

将数据结果转换为矩阵

# [np, 矩阵基础](https://www.jianshu.com/p/9bb94f6ca1b0)

https://www.cnblogs.com/xzcfightingup/p/7598293.html

矩阵生成：http://www.zmonster.me/2016/02/25/creation-and-io-of-ndarray.html

```python
np.ones((3, 4))

[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]

```



# [np.ndarray.tolist](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.tolist.html)

将数组转换为list

```python
>>> a = np.array([1, 2])
>>> a.tolist()
[1, 2]
>>> a = np.array([[1, 2], [3, 4]])
>>> list(a)
[array([1, 2]), array([3, 4])]
>>> a.tolist()
[[1, 2], [3, 4]]
```

# [numpy.linalg](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html)

numpy下的linalg=linear+algebra，包含很多线性代数的运算，主要用法有以下几种

https://blog.csdn.net/chunjing6629/article/details/80554778



# [python.tkinter](http://www.runoob.com/python/python-gui-tkinter.html)

python的GUI制作库

check button 复选框http://www.runoob.com/python/python-tk-checkbutton.html



# [矩阵的转置](https://blog.csdn.net/Asher117/article/details/82934857)

随后再看

T https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.T.html

transpose https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.transpose.html



# [numpy.matrix.getA()](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.matrix.getA.html)

把矩阵转换为ndarray



# [删除元素](https://blog.csdn.net/deqiangxiaozi/article/details/75808863)

del 不适用array

```python
# del
a = ['a','b','c']
del a[0]  #指定删除0位的元素
print(a)
输出为：['b', 'c']

# remove
a = ['a','b','c']
a.remove('b') #删除指定元素
print(a)
输出为：['a', 'c']

# pop
a = ['a','b','c']
b = ['d','e','f']
# pop的命令，其有返回值，可赋值带出
c = a.pop() #默认删除-1位置元素'c',并将删除元素返回值赋值
d = b.pop(0) #删除0位元素'd',并将删除元素返回值赋值
print(a,b,c,d)
返回值：['a', 'b'] ['e', 'f'] c d
```

# [sys](https://www.cnblogs.com/Archie-s/p/6860301.html)
https://docs.python.org/3/library/sys.html
sys.path.append("自定义模块路径")

# [os](https://www.cnblogs.com/yufeihlf/p/6179547.html)
http://www.runoob.com/python3/python3-os-file-methods.html
os.getwcd()查看当前所在的路径
os.path.split()将路径分解为(文件夹,文件名)，返回的是元组类型
os.path.join()组合成路径
os.open()打开文件
os.chdir()改变工作路径

# [csv](https://www.cnblogs.com/pyxiaomangshe/p/8026483.html)
```python
reader(csvfile, dialect='excel', **fmtparams)
csvfile，必须是支持迭代(Iterator)的对象，可以是文件(file)对象或者列表(list)对象，如果是文件对
象，打开时需要加"b"标志参数。
 
dialect，编码风格，默认为excel的风格，也就是用逗号（,）分隔，dialect方式也支持自定义，通过调用register_dialect方法来注册，下文会提到。
 
fmtparam，格式化参数，用来覆盖之前dialect对象指定的编码风格。
```
```python
import csv
with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
```
# [json](http://www.runoob.com/python/python-json.html)
```python
json.dumps(obj, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, encoding="utf-8", default=None, sort_keys=False, **kw)

json.loads(s[, encoding[, cls[, object_hook[, parse_float[, parse_int[, parse_constant[, object_pairs_hook[, **kw]]]]]]]])
```

# [python3启动虚拟环境](https://blog.csdn.net/lose_812/article/details/79851677)
```shell
$ mkdir work
$ python3 -m venv work/

# 查看虚拟环境目录结构
$ cd work
$ ls
bin  include  lib  lib64  pyvenv.cfg  share
$ cd work
$ source bin/activate

# 前面有括号xx，即表示虚拟环境激活成功，在激活环境下pip安装的包单独管理，不会冲突 
(work) $
# 退出虚拟环境
(work) $ deactivate
```

# [pandas apply applymap map](https://blog.csdn.net/u010814042/article/details/76401133/)
总的来说就是apply()是一种让函数作用于列或者行操作，applymap()是一种让函数作用于DataFrame每一个元素的操作，而map是一种让函数作用于Series每一个元素的操作

# [时间数据分箱 resample]（http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html）

# [获取最大值的索引 idxmax](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.idxmax.html)

# [np.random 的使用]（https://blog.csdn.net/mengenqing/article/details/80615668）

# [np.flatten ravel](https://blog.csdn.net/liuweiyuxiang/article/details/78220080)
# [数据分箱](https://zhuanlan.zhihu.com/p/31168548)
```python
>>> import numpy as np
>>> import pandas as pd
>>> score_list = np.random.randint(30,100,size=20)
>>> socre_list
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'socre_list' is not defined
>>> score_list
array([98, 74, 91, 49, 78, 46, 73, 61, 45, 56, 57, 63, 92, 53, 63, 86, 35,
       60, 63, 88])
>>> bins = [0, 59, 70, 80, 100]
>>> score_cat = pd.cut(score_list, bins)
>>> pd.value_counts(score_cat)
(0, 59]      7
(80, 100]    5
(59, 70]     5
(70, 80]     3
dtype: int64
>>> df['score'] = score_list
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'df' is not defined
>>> df = pd.DataFrame()
>>> df['score'] = score_list
>>> df['student'] = [pd.util.testing.rands(3) for i in range(20)]
>>> df['Categories'] = pd.cut(df['score'], bins, labels=['Low', 'OK', 'Good', 'Great'])
>>> df
    score student Categories
0      98     QFG      Great
1      74     mbF       Good
2      91     RRe      Great
3      49     v31        Low
4      78     sK9       Good
5      46     8wR        Low
6      73     JQt       Good
7      61     RME         OK
8      45     dYT        Low
9      56     i18        Low
10     57     5s0        Low
11     63     Y22         OK
12     92     CR7      Great
13     53     O3u        Low
14     63     IgN         OK
15     86     VP5      Great
16     35     XJU        Low
17     60     FdV         OK
18     63     DLm         OK
19     88     YxF      Great
```

# joblib多线程
https://joblib.readthedocs.io/en/latest/

# 随机生成字符串
https://www.cnblogs.com/zqifa/p/python-random-1.html

# 装饰器@
https://www.cnblogs.com/cccy0/p/8799491.html

# python类的可见与不可见
https://blog.csdn.net/u012279631/article/details/81363929

# cls 和 self
https://blog.csdn.net/daijiguo/article/details/78499422

# python的异常处理
https://segmentfault.com/a/1190000007736783
 
# Lazy evalutaion p39（高效python）
1. 避免不必要的计算: or表达式，可能性高的True放在前面，and则放在后面
提升50%
```python
if w == b and w == c:

```
2. 节省空间
生成器的使用: yield
yield的使用：https://blog.csdn.net/mieleizhi0522/article/details/82142856
功能等同于return，只是执行流程不同，是一步一步的执行。使用next执行下一个参数在生成器中的运行，send是传参用的。

# 使用工厂函数 保证类型符合要求

# is 和 ==
is 用来判断存储空间是否一致
== 用来判断相等
