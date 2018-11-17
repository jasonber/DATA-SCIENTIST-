# TypeError: cannot compare a dtyped [int64] array with a scalar of type [bool]

```python
# 错误
chipo[chipo['item_name'] == 'Canned Soda' & chipo['quantity'] > 1]['order_id'].count()

# 正确
chipo[(chipo['item_name'] == 'Canned Soda') & (chipo['quantity'] > 1)]['order_id'].count()
```

# TypeError: __call__() takes from 1 to 2 positional arguments but 3 were given

```python
#错误
army.iloc(['Maine', 'Alaska'], ['deaths', 'size', 'deserters'])
#正确
army.loc[['Maine', 'Alaska'], ['deaths', 'size', 'deserters']] 
```

# TypeError: cannot perform reduce with flexible type

```python
#错误
army.iloc[['Maine', 'Alaska'], ['deaths', 'size', 'deserters']]
#正确
army.loc[['Maine', 'Alaska'], ['deaths', 'size', 'deserters']] 
```

# ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

```python
# 错误
army[(army['deaths'] > 500) or (army['deaths'] < 50)
# 正确
army[(army['deaths'] > 500) | (army['deaths'] < 50)
```

区分位运算与逻辑运算

# TypeError: 'method' object is not subscriptable

```python
# 错误
users.groupby['occupation']['age'].mean()
# 正确
users.groupby('occupation')['age'].mean()
```



# TypeError: reduction operation 'argmax' not allowed for this dtype

说明值的数据类型不是数值



# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 28: invalid start byte

```python
# 报错
Past=pd.read_csv("C:/Users/Admin/Desktop/Python/Past.csv",encoding='utf-8')
# 修正
Past=pd.read_csv("C:/Users/Admin/Desktop/Python/Past.csv",encoding='cp1252')
```



读取文档的时候，utf-8无法解码



# [python3 TypeError: 'map' object is not subscriptable](https://blog.csdn.net/mingyuli/article/details/81238858)

In Python 3, map returns an iterable object of type map, and not a subscriptible list, which would allow you to write map[i]. To force a list result, write

https://www.e-learn.cn/content/wangluowenzhang/87273

```python
# 报错
d = map(apply_filters_to_token, sentences)
x.append(d)
# 正确
d = list(map(apply_filters_to_token, sentences))
x.append(d)
```

# ValueError: not enough values to unpack (expected 3, got 2)

函数返回值少于要求的值

# TypeError: 'range' object doesn't support item deletion

python3.x , 出现错误 'range' object doesn't support item deletion

原因：python3.x   range返回的是range对象，不返回数组对象

解决方法：把 trainingSet = range(50) 改为 trainingSet = list(range(50))
