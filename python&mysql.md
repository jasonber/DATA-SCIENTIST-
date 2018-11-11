# 设置mariadb

https://blog.csdn.net/zhezhebie/article/details/73549741

# 功能：获取数据库中的所有表的字段（列名）

```python
def table_columns(database):
    from datetime import datetime
    start_time = datetime.now()

    def get_columns(table, cnn):
        query = "show  columns from {};".format(table)
        try:
            df = pd.read_sql(query, cnn)
            print("success")
            return df
        except Exception as e:
            print(e)

    try:
        table_name = pd.read_sql("show tables;", cnn)
    except Exception as e:
        print(e)

    list_of_table = []
    create_var = locals()
    for i, j in zip(table_name.iloc[:, 0], table_name.iloc[:, 0]):
        create_var[j] = get_columns(i, cnn)
        list_of_table.append(create_var[j])

    column_dict = {}
    for i, j in zip(table_name.iloc[:, 0], range(0, 8)):
        column_dict[i] = list_of_table[j].iloc[:, 0]

    columns_of_table = pd.DataFrame.from_dict(column_dict)
    end_time = datetime.now()
    print('time of this process: {}s'.format((end_time - start_time).seconds))
    return columns_of_table

```

