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

# ORDER BY

```mysql
SELECT column_name,column_name
FROM table_name
ORDER BY column_name,column_name ASC|DESC;
```

# 找出所有的1970获奖者，按照subject和获奖者姓名排序，其中Economic和Chemistry 按照升序排列在最后

```mysql
SELECT *
FROM nobel_win
WHERE year=1970 
ORDER BY
 CASE
    WHEN subject IN ('Economics','Chemistry') THEN 1
    ELSE 0
 END ASC,
 subject,
 winner;

```

[case](https://www.cnblogs.com/4littleProgrammer/p/4820006.html)

1、case表达式会从最初的when子句中的 判断表达式进行判断。如果为真，就返回then子句中的表达式，case表达式的执行到此结束

2、else null，指定了不满足when子句条件的操作

3、end 必须写，end后面可加入表达式，表示对case的操作

# [聚合函数](https://www.cnblogs.com/ghost-xyx/p/3811036.html)

SQL中提供的聚合函数可以用来统计、求和、求最值等等。

分类：

–COUNT：统计行数量
–SUM：获取单个列的合计值
–AVG：计算某个列的平均值
–MAX：计算列的最大值
–MIN：计算列的最小值

[count](http://www.w3school.com.cn/sql/sql_func_count.asp)

SQL COUNT(\*) 语法
COUNT(\*) 函数返回表中的记录数

```sql
SELECT COUNT(*) FROM table_name
```



# 查找最便宜的商品

```mysql
SELECT pro_name, pro_price
   FROM item_mast
   WHERE pro_price = 
    (SELECT MIN(pro_price) FROM item_mast);
```

sql的赋值方式？ “=”的意义？

# Write a SQL query to display those customers who are neither belongs to the city New York nor grade value is more than 100. 

```mysql
SELECT * 
FROM customer 
WHERE NOT (city = 'New York' OR grade>100);

等价
SELECT *
FROM customer
WHERE (city != 'New York' AND grade <= 100);

不等价
SELECT *
FROM customer
WHERE city != 'New York' AND grade <= 100;
```

# Write a SQL statement to display either those orders which are not issued on date 2012-09-10 and issued by the salesman whose ID is 505 and below or those orders which purchase amount is 1000.00 and below.

```mysql
SELECT * 
FROM  orders 
WHERE NOT ((ord_date ='2012-09-10' 
AND salesman_id>505) 
OR purch_amt>1000.00);
```



# Write a SQL statement to display salesman_id, name, city and commission who gets the commission within the range more than 0.10% and less than 0.12%.

```mysql
SELECT salesman_id,name,city,commission 
FROM salesman 
WHERE (commission > 0.10 
AND commission< 0.12);

不等价
SELECT *
FROM salesman
WHERE commission BETWEEN 0.10 AND 0.12;
```

BETWEEN包含等号

# Write a SQL statement where i) order dates are anything but 2012-08-17, or customer id is not greater than 3005 ii) and purchase amount is not below 1000.

```mysql
SELECT * 
FROM  orders 
WHERE NOT((ord_date ='2012-08-17' 
OR customer_id>3005) 
AND purch_amt<1000);

不等价
SELECT *
FROM orders
WHERE (NOT(ord_date = '2012-08-17' OR customer_id > 3005) 
       AND NOT purch_amt < 1000);
```

筛选的条件要屡清楚

# Write a SQL query to display order number, purchase amount, achived, the unachieved percentage for those order which exceeds the 50% of the target value of 6000.

```mysql
SELECT ord_no,purch_amt, 
(100*purch_amt)/6000 AS "Achieved %", 
(100*(6000-purch_amt)/6000) AS "Unachieved %" 
FROM  orders 
WHERE (100*purch_amt)/6000>50;
```



# *Write a SQL query to display order number, purchase amount, achived, the unachieved percentage for those order which exceeds the 50% of the target value of 6000.*

```mysql
SELECT ord_no,purch_amt, 
(100*purch_amt)/6000 AS "Achieved %", 
(100*(6000-purch_amt)/6000) AS "Unachieved %" 
FROM  orders 
WHERE (100*purch_amt)/6000>50;

# 哪里错了？除号写错了
SELECT ord_no, purch_amt, 
(purch_amt\6000)*100 AS "Achieved %",
(1-(purch_amt\6000))*100 AS "Unachieved %"
FROM orders
WHERE (purch_amt/6000)*100 > 50;
```

alias的用法 

# Write a query in SQL to find the data of employees whose last name is Dosni or Mardy.

```mysql
SELECT * 
 FROM emp_details
  WHERE emp_lname ='Dosni' OR emp_lname= 'Mardy';

等价
SELECT * 
FROM emp_details
WHERE emp_lname IN ('Dosni', 'Mardy'); # 也适用于数值
```



# Write a query to filter all those orders with all information which purchase amount value is within the range 500 and 4000 except those orders of purchase amount value 948.50 and 1983.43.

```mysql
SELECT * 
FROM orders 
WHERE (purch_amt BETWEEN 500 AND 4000) 
AND NOT purch_amt IN(948.50,1983.43);

# 这是什么鬼。。。
SELECT *
FROM orders
WHERE
  CASE
  WHEN purch_amt IN ('948.50', '1983.43') THEN 0
  ELSE NULL
  END
 purch_amt BETWEEN 500 AND 4000;
```

# Write a SQL statement to find those salesmen with all other information and name started with any latter within 'A' and 'K'.

```mysql
SELECT *
FROM salesman
WHERE name BETWEEN 'A' and 'L';
# 不用指明 start with吗？
```

# *SQL的 ""和''的作用*

https://blog.csdn.net/u010566813/article/details/51864375

标准sql中不存在双引号。如果字符中有单引号，那么sql中用两个单引号来表示。

双引号“ ”表示输入的字符，比如别名

```sql
SELECT AVG(pro_price) AS "Average Price"
FROM item_mast;
```



# Write a SQL statement to find those rows from the table testtable which contain the escape character underscore ( _ ) in its column 'col1'.

```mysql
SELECT *
FROM testtable
WHERE col1 LIKE '%/_%' ESCAPE '/';
```

[escape](https://blog.csdn.net/david_520042/article/details/6909230) 转义符 把通配符 转换成 普通符

/只是一个记号，标识escape起转换作用的位置。只转换标识后的通配符。

# IS NULL 

否定式

```mysql
col IS NOT NULL
NOT col IS NULL
```

# Write a SQL statement to find the number of salesmen currently listing for all of their customers

```mysql
SELECT COUNT (DISTINCT salesman_id) 
FROM orders;
```



# [having 和 where的区别](https://blog.csdn.net/qmhball/article/details/7941638)

Where 是一个约束声明，使用Where约束来自数据库的数据，Where是在结果返回之前起作用的，Where中不能使用聚合函数。
Having是一个过滤声明，是在查询返回结果集以后对查询结果进行的过滤操作，在Having中可以使用聚合函数。
在查询过程中聚合语句(sum,min,max,avg,count)要比having子句优先执行。

而where子句在查询过程中执行优先级高于聚合语句。

having的用处

http://www.w3school.com.cn/sql/sql_having.asp

having的用法：前面必须有group by

https://www.dofactory.com/sql/having

```mysql
SELECT column-names
  FROM table-name
 WHERE condition
 GROUP BY column-names
HAVING condition
```

HAVING filters records that work on summarized GROUP BY results.
HAVING applies to summarized group records, whereas WHERE applies to individual records.
Only the groups that meet the HAVING criteria will be returned.
HAVING requires that a GROUP BY clause is present.
WHERE and HAVING can be in the same query.



# Write a SQL statement to find the highest purchase amount with their ID and order date, for those customers who have a higher purchase amount in a day is within the range 2000 and 6000.

```mysql
SELECT customer_id,ord_date,MAX(purch_amt) 
FROM orders 
GROUP BY customer_id,ord_date 
HAVING MAX(purch_amt) BETWEEN 2000 AND 6000;
```



如何选出某一极值的所有信息？

# [GROUP BY](http://www.w3school.com.cn/sql/sql_groupby.asp)

```sql
SELECT column_name, aggregate_function(column_name)
FROM table_name
WHERE column_name operator value
GROUP BY column_name
```

# Write a SQL statement to make a report with customer ID in such a manner that, the largest number of orders booked by the customer will come first along with their highest purchase amount.

```sql
SELECT customer_id, COUNT(DISTINCT ord_no), 
MAX(purch_amt) 
FROM orders 
GROUP BY customer_id 
ORDER BY 2 DESC;
```

2代表第二列？

# [多表取数](https://www.w3resource.com/sql-exercises/sql-exercises-quering-on-multiple-table.php)

https://blog.csdn.net/wmz545546/article/details/77921550

# Write a query to find those customers with their name and those salesmen with their name and city who lives in the same city.

```sql
SELECT customer.cust_name,
salesman.name, salesman.city
FROM salesman, customer
WHERE salesman.city = customer.city;
```

# Write a SQL statement to display all those orders by the customers not located in the same cities where their salesmen live.

```sql
SELECT ord_no, cust_name, orders.customer_id, orders.salesman_id
FROM salesman, customer, orders
WHERE customer.city <> salesman.city
AND orders.customer_id = customer.customer_id
AND orders.salesman_id = salesman.salesman_id;
```

不等号<>

# Write a SQL statement that shorts out the customer and their grade who made an order. Each of the customers must have a grade and served by at least a salesman, who belongs to a city.

```sql
SELECT customer.cust_name AS "Customer",
customer.grade AS "Grade"
FROM orders, salesman, customer
WHERE orders.customer_id = customer.customer_id
AND orders.salesman_id = salesman.salesman_id
AND salesman.city IS NOT NULL
AND customer.grade IS NOT NULL;
```

