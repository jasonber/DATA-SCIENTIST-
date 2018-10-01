# 展现字符串
```mysql
select 'This is SQL Exercise, Practice and Solution';
```

# 用3列展现3个数字

```mysql
select 1, 2, 3;
```

# 求两个数字之和

```mysql
select 10 + 15;
```

# 求一个算式

```mysql
select 10 + 15 - 5 * 2;
```

# sql语句的两个顺序

语法顺序：

```mysql
Select 
From 
Where LIKE BETWEEN IN
Group by 
Having 
Order by 
Limit
```

执行顺序：

```mysql
From 
Where 
Group by 
Having 
Select 
Order by 
Limit
```

# [Between](http://www.w3school.com.cn/sql/sql_between.asp)
``` mysql
SELECT 
FROM
WHERE
BETWEEN     AND     
```

等价于

```mysql
SELECT 
  FROM 
 WHERE subject = 'Chemistry'
   AND year>=1965 AND year<=1975;
```

# [IN](http://www.w3school.com.cn/sql/sql_in.asp)

IN 操作符允许我们在 WHERE 子句中规定多个值。

```mysql
SELECT column_name(s)
FROM table_name
WHERE column_name IN (value1,value2,...)
```

# 选择姓Louis的人的信息

```mysql
SELECT *
 FROM nobel_win 
   WHERE winner LIKE 'Louis%';
```

# [LIKE](http://www.w3school.com.cn/sql/sql_like.asp)

LIKE 操作符用于在 WHERE 子句中搜索列中的指定模式。

```mysql
SELECT column_name(s)
FROM table_name
WHERE column_name LIKE pattern
```

"%" 可用于定义通配符（模式中缺少的字母）。

[LIKE配套的通配符](http://www.w3school.com.cn/sql/sql_wildcards.asp)

```
% 替代一个或多个字符
_下划线 仅替代一个字符

[字符列] 字符列中的任何单一字符

[!字符列][^字符列]不存在字符列中的任何单一字符 
```

# Write a SQL query to show all the winners in Physics for 1970 together with the winner of Economics for 1971.

```mysql
SELECT *
 FROM nobel_win 
   WHERE (subject ='Physics' AND year=1970)
UNION
SELECT *
 FROM nobel_win 
     WHERE (subject ='Economics' AND year=1971);
```

# [UNION](http://www.w3school.com.cn/sql/sql_union.asp)

UNION 操作符用于合并两个或多个 SELECT 语句的结果集。

请注意，UNION 内部的 SELECT 语句必须拥有相同数量的列。列也必须拥有相似的数据类型。同时，每条 SELECT 语句中的列的顺序必须相同。

```mysql
SELECT column_name(s) FROM table_name1
UNION
SELECT column_name(s) FROM table_name2
```

默认地，UNION 操作符选取不同的值。如果允许重复的值，请使用 UNION ALL

```mysql
SELECT column_name(s) FROM table_name1
UNION ALL
SELECT column_name(s) FROM table_name2
```

()的作用是啥？