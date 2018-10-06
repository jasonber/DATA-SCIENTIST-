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



