# TypeError: cannot compare a dtyped [int64] array with a scalar of type [bool]

```python
# 错误
chipo[chipo['item_name'] == 'Canned Soda' & chipo['quantity'] > 1]['order_id'].count()

# 正确
chipo[(chipo['item_name'] == 'Canned Soda') & (chipo['quantity'] > 1)]['order_id'].count()
```

