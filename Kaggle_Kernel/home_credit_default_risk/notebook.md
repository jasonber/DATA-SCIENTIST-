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