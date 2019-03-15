# sql的编写理解
1. 就是从表中按照条件取得数据
2. 取数的信息：
      1. 取什么字段 
      2. 从什么样的表中取
      3. 数据应该满足什么样的要求
3. 两种取数思路：
   1. 建好包含所取信息的表，按照条件取数
      （子查询，join建表）
   2. 按照条件获取字段，在现有的表中取出这些字段
      （where in , like ,regep)

# sql语句的两个顺序

1. 语法顺序：S F W G H O L

```sql
Select 1
From 2 
Where LIKE BETWEEN IN 3
Group by 4
Having 5
Order by 6 
Limit 7 Offset 
```

2. 执行顺序：F W G H S O L

```sql
From 2
Where 3
Group by 4 
Having 5
Select 1
Order by 6 
Limit 7
```

# mysql的取数过程 
是一个循环。遍历所有表看看是否满足where的条件。按照where的条件，制作出一个表，在表内进行分组，筛选，然后再用select 从表中提取字段。这时会获得一个目标表，在这个表上进行排序等操作。

# 展现字符串
```mysql
select 'This is SQL Exercise, Practice and Solution';
```
# 关键词的意义
select 选啥 from 从哪选（子查询、联结表）Where 啥条件 group by 咋分组
子查询可出现在任何位置

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

Between And 在不同的数据库中，边界会有不同

# [IN](http://www.w3school.com.cn/sql/sql_in.asp)

IN 操作符允许我们在 WHERE 子句中规定多个值。

```mysql
SELECT column_name(s)
FROM table_name
WHERE column_name IN (value1,value2,...)
```

# 选择姓Louis的人的信息

```sql
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
# 否定
NOT LIKE
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



# ORDER BY

```mysql
SELECT column_name,column_name
FROM table_name
ORDER BY column_name,column_name ASC|DESC;
ORDER BY 列号
```

# *记住* 找出所有的1970获奖者，按照subject和获奖者姓名排序，其中Economic和Chemistry 按照升序排列在最后


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
case when 实例 https://www.cnblogs.com/aipan/p/7770611.html

1、case表达式会从最初的when子句中的 判断表达式进行判断。如果为真，就返回then子句中的表达式，case表达式的执行到此结束

2、else null，指定了不满足when子句条件的操作

3、end 必须写，end后面可加入表达式，表示对case的操作

4、case 字段 when 只能进行转换， case when 条件 能对字段进行条件赋值以及可以对字段进行运算

https://www.cnblogs.com/Richardzhu/p/3571670.html

case when 可以看做是对数据的行列转换

实例

[Write a query to display the employee id, name ( first name and last name ) and the job id column with a modified title SALESMAN for those employees whose job title is ST_MAN and DEVELOPER for whose job title is IT_PROG.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-23.php)

```sql
--简单case函数
SELECT  employee_id,  first_name, last_name,  
CASE job_id  
WHEN 'ST_MAN' THEN 'SALESMAN'  
WHEN 'IT_PROG' THEN 'DEVELOPER'  
ELSE job_id  
END AS designation,  salary 
FROM employees;
```

[Write a query to display the employee id, name ( first name and last name ), salary and the SalaryStatus column with a title HIGH and LOW respectively for those employees whose salary is more than and less than the average salary of all employees.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-24.php)

```sql
--搜索case函数
SELECT  employee_id,  first_name, last_name, salary,  
CASE WHEN salary >= 
(SELECT AVG(salary) 
FROM employees) THEN 'HIGH'  
ELSE 'LOW'  
END AS SalaryStatus 
FROM employees;
```

# 单双引号
‘’表示表内的值
“”表示文本

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

结合就join 组合查询。

# 查找最便宜的商品

```mysql
SELECT pro_name, pro_price
   FROM item_mast
   WHERE pro_price = 
    (SELECT MIN(pro_price) FROM item_mast);
```

sql的赋值方式？ “=”的意义？
= 号的意义是比较，：=的意义是赋值

# [Write a SQL query to display those customers who are neither belongs to the city New York nor grade value is more than 100.](https://www.w3resource.com/sql-exercises/sql-boolean-operator-exercise-5.php)

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
[多条件查询](https://blog.csdn.net/liuying_1001/article/details/37700883)
注意：
1、逻辑符号的等级， and优先级高于or
2、条件的意义      

# [Write a SQL statement to display either those orders which are not issued on date 2012-09-10 and issued by the salesman whose ID is 505 and below or those orders which purchase amount is 1000.00 and below.](https://www.w3resource.com/sql-exercises/sql-boolean-operator-exercise-6.php)

```mysql
SELECT * 
FROM  orders 
WHERE NOT ((ord_date ='2012-09-10' 
AND salesman_id>505) 
OR purch_amt>1000.00);
```



# [Write a SQL statement to display salesman_id, name, city and commission who gets the commission within the range more than 0.10% and less than 0.12%.](https://www.w3resource.com/sql-exercises/sql-boolean-operator-exercise-7.php)

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

BETWEEN包含等号,这个要注意 各个语言是不一样的

# [Write a SQL statement where i) order dates are anything but 2012-08-17, or customer id is not greater than 3005 ii) and purchase amount is not below 1000.](https://www.w3resource.com/sql-exercises/sql-boolean-operator-exercise-9.php)

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

# [Write a SQL query to display order number, purchase amount, achived, the unachieved percentage for those order which exceeds the 50% of the target value of 6000.](https://www.w3resource.com/sql-exercises/sql-boolean-operator-exercise-10.php)
[运算符的顺序](https://docs.microsoft.com/zh-cn/sql/t-sql/language-elements/operator-precedence-transact-sql?view=sql-server-2017)

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

什么时候用Having啊？？有点懵

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



# [字符串拼接](http://www.cnblogs.com/rainman/p/6203065.html)

concat , +, '||' 跟excel很像



# Write a query in SQL to display job ID for those jobs that were done by two or more for more than 300 days.

```mysql
# 错误
SELECT job_id
FROM job_history
GROUP BY job_id
HAVING COUNT(end_date - start_date > 300) > 2;

# 正确
SELECT job_id 
	FROM job_history 
		WHERE end_date-start_date >300 
			GROUP BY job_id 
				HAVING COUNT(*)>=2;
```



# [group by注意事项](https://blog.csdn.net/haiross/article/details/50440176)

https://www.cnblogs.com/chenleiustc/archive/2009/07/30/1535042.html

ORDER BY:当使用ORDER BY子句时，多数情况下指定的排列序列都是选择列，但是排序列也可以不是选择列。但是如果在SELECT子句中使用了DISTINCT关键字，则排序列就必须是选择列了，否则会报错。

GROUP BY：告诉数据库如何将查询出的数据进行分组，然后数据库才知道将组处理函数作用于已经分好的组。
注意点：
1、组处理函数只能出现在选择列表,ORDER BY子句，HAVING子句中，而不能出现在WHERE子句和GROUP BY子句中
2、除了COUNT(*)之外，其他组处理函数都会忽略NULL行
3、如果选择列表同时包含列，表达式和组函数，则这些列，表达式都必须出现在GROUP BY子句中
4、在组处理函数中可以指定ALL,DISTINCT选项。其中ALL是默认的选项，表示统计所有的行（包括重复的行），而DISTINCT只会统计不同的行



# Write a query in SQL to display those departments where any manager is managing 4 or more employees.

```mysql
# 错误
SELECT DISTINCT department_id
FROM employees
GROUP BY manager_id, department_id
HAVING COUNT(employee_id) >= 4;

# 正确
SELECT DISTINCT department_id
	FROM employees
		GROUP BY department_id, manager_id 
			HAVING COUNT(employee_id) >=4;
```

分组也是有顺序的



# [DISTINCT](http://www.runoob.com/sql/sql-distinct.html)

在表中，一个列可能会包含多个重复值，有时您也许希望仅仅列出不同（distinct）的值。

DISTINCT 关键词用于返回唯一不同的值。



# 多表查询

Write a SQL statement to make a list with order no, purchase amount, customer name and their cities for those orders which order amount between 500 and 2000.

```mysql
SELECT orders.ord_no, orders.purch_amt, customer.cust_name, customer.city 
FROM orders
JOIN customer
ON orders.customer_id = customer.customer_id 
WHERE orders.purch_amt BETWEEN 500 AND 2000;
# 等价
SELECT  a.ord_no,a.purch_amt,
b.cust_name,b.city 
FROM orders a,customer b 
WHERE a.customer_id=b.customer_id 
AND a.purch_amt BETWEEN 500 AND 2000;
```



# [三个表及以上的连接](https://blog.csdn.net/zht666/article/details/8555164)

```sql
SELECT * 
FROM 表1 
INNER JOIN 表2 
ON 表1.字段号=表2.字段号 
INNER JOIN 表3 
ON 表1.字段号=表3.字段号) 
INNER JOIN 表4 
ON 表1.字段号=表4.字段号
```

# [NATURAL JOIN](https://blog.csdn.net/aeolus_pu/article/details/7789543)

​        自然连接（NATURAL JOIN）是一种特殊的等价连接，它将表中具有相同名称的列自动进行记录匹配。自然连接不必指定任何同等连接条件。图9.9给出了典型的自然连接示意图。自然连接自动判断相同名称的列，而后形成匹配。缺点是，虽然可以指定查询结果包括哪些列，但不能人为地指定哪些列被匹配。另外，自然连接的一个特点是连接后的结果表中匹配的列只有一个。

Write a SQL statement to make a join on the tables salesman, customer and orders in such a form that the same column of each table will appear once and only the relational rows will come.

```sql
SELECT * 
FROM orders 
NATURAL JOIN customer  
NATURAL JOIN salesman;
```

# [FULL OUTER JOIN](http://www.runoob.com/sql/sql-join-full.html)

http://www.w3school.com.cn/sql/sql_join_full.asp 例子

Write a SQL statement to make a report with customer name, city, order no. order date, purchase amount for only those customers on the list who must have a grade and placed one or more orders or which order(s) have been placed by the customer who is neither in the list not have a grade.

```sql
SELECT a.cust_name,a.city, b.ord_no,
b.ord_date,b.purch_amt AS "Order Amount" 
FROM customer a 
FULL OUTER JOIN orders b 
ON a.customer_id=b.customer_id 
WHERE a.grade IS NOT NULL;
```



FULL OUTER JOIN 关键字只要左表（table1）和右表（table2）其中一个表中存在匹配，则返回行.

FULL OUTER JOIN 关键字结合了 LEFT JOIN 和 RIGHT JOIN 的结果。

mysql 中没有FULL OUTER JOIN 那该怎么实现呢？

# [CROSS JOIN](http://www.cnblogs.com/chenxizhang/archive/2008/11/10/1330325.html)

```sql
SQL CROSS JOIN syntax:
SELECT * FROM [TABLE 1] CROSS JOIN [TABLE 2]
-- 等价
SELECT * FROM [TABLE 1], [TABLE 2]  //重点,平时写法要注意啊
```

Write a SQL statement to make a cartesian product between salesman and customer i.e. each salesman will appear for all customer and vice versa.

cartesian product 笛卡尔积

```sql
SELECT * 
FROM salesman a 
CROSS JOIN customer b;
```



# [Write a SQL query to display the name of each company along with the ID and price for their most expensive product.](https://www.w3resource.com/sql-exercises/sql-joins-exercise-25.php)

```sql
SELECT A.pro_name, A.pro_price, F.com_name
   FROM item_mast A INNER JOIN company_mast F
   ON A.pro_com = F.com_id
     AND A.pro_price =
     (
       SELECT MAX(A.pro_price)
         FROM item_mast A
         WHERE A.pro_com = F.com_id
     );
```

# [LEFT RIGHT 是怎么判断以那个表格为标准的？](http://blog.sina.com.cn/s/blog_161231e190102wqog.html)

交集交的是什么？并集并的是什么？

join 的是列名，要显示哪些行，用inner right left 来判断。

带有B中什么信息的A： 要显示的是A的行。

```sql
SELECT E.department_id AS "E-department", D.department_id AS "D-department",
E.first_name, E.last_name, 
L.city, L.state_province
FROM employees E
RIGHT JOIN departments D -- E与D以department_id为条件的交集，以及D的全部department_id
ON E.department_id = D.department_id
INNER  JOIN locations L
ON D.location_id = L.location_id;
```

Left join：即左连接，是以左表为基础，根据ON后给出的两表的条件将两表连接起来。结果会将左表所有的查询信息列出，而右表只列出ON后条件与左表满足的部分。左连接全称为左外连接，是外连接的一种。

**左右表的条件交集+左表该条件的全部**

Right join：即右连接，是以右表为基础，根据ON后给出的两表的条件将两表连接起来。结果会将右表所有的查询信息列出，而左表只列出ON后条件与右表满足的部分。右连接全称为右外连接，是外连接的一种。

**左右表的条件交集+右表该条件的全部**

Inner join：即内连接，同时将两表作为参考对象，根据ON后给出的两表的条件将两表连接起来。结果则是两表同时满足ON后的条件的部分才会列出。 

**交集**

CROSS join：笛卡尔积，全部数据结合在一起，不是乘除，而是位置的一种排列方法。前面的所有结合方式都是在此基础上完成的。



# 7. Write a query in SQL to display the first and last name and salary for those employees who earn less than the employee earn whose number is 182.

```sql
SELECT E.first_name, E.last_name, E.salary 
  FROM employees E 
   JOIN employees S
     ON E.salary < S.salary 
      AND S.employee_id = 182;
```

# [8. Write a query in SQL to display the first name of all employees including the first name of their manager.](https://www.w3resource.com/sql-exercises/joins-hr/sql-joins-hr-exercise-8.php)

有管理者的员工的名字以及这位管理者的名字

```sql
--错误
SELECT E.first_name AS "Employee Name", 
M.first_name AS "Manager Name"
FROM employees E
LEFT JOIN employees M
ON E.employee_id = M.manager_id;--用整个公司的员工去匹配管理者，且返回所有员工id

--正确
SELECT E.first_name AS "Employee Name", 
   M.first_name AS "Manager"
     FROM employees E 
       JOIN employees M
         ON E.manager_id = M.employee_id;

--等价
SELECT E.first_name AS "Employee Name", 
M.first_name AS "Manager Name"
FROM employees E
LEFT JOIN employees M
ON M.employee_id = E.manager_id;--用整个公司的管理者去匹配公司的员工，返回员工的管理者id
```



# 11. Write a query in SQL to display the first name of all employees and the first name of their manager including those who does not working under any manager.

```sql
--原题答案
SELECT E.first_name AS "Employee Name",
   M.first_name AS "Manager"
    FROM employees E 
      LEFT OUTER JOIN employees M
       ON E.manager_id = M.employee_id;

--选出没有管理者的员工姓名
--错误答案：因为条件的筛选在join之前，
SELECT E.first_name AS "Employees", 
M.first_name AS "Manager"
FROM employees E
LEFT JOIN employees M
ON E.manager_id = M.employee_id
And M.manager_id IS NULL;
--正确
SELECT first_namen
FROM employees E
WHERE manager_id  = 0;
```

# [JOIN USING](https://stackoverflow.com/questions/13750152/using-keyword-vs-on-clause-mysql)

using用于有相同列名的多表连接。

# *[21. Write a query in SQL to display the country name, city, and number of those departments where at leaste 2 employees are working.](https://www.w3resource.com/sql-exercises/joins-hr/sql-joins-hr-exercise-21.php)*

看不懂的子查询

```sql
--错误
SELECT C.country_name, 
L.city, 
COUNT(D.department_id)
FROM departments D
LEFT JOIN locations L
ON D.location_id = L.location_id
LEFT JOIN countries C
ON L.country_id = C.country_id
LEFT JOIN employees E
ON D.department_id = E.department_id
GROUP BY D.department_id
HAVING COUNT(E.employee_id) >= 2;

--正确
SELECT country_name,city, COUNT(department_id)
	FROM countries 
		JOIN locations USING (country_id) 
		JOIN departments USING (location_id) 
WHERE department_id IN 
    (SELECT department_id 
		FROM employees 
	 GROUP BY department_id 
	 HAVING COUNT(department_id)>=2)
GROUP BY country_name,city;
```



# *[25. Write a query in SQL to display full name(first and last name), job title, starting and ending date of last jobs for those employees with worked without a commission percentage.](https://www.w3resource.com/sql-exercises/joins-hr/sql-joins-hr-exercise-25.php)*

```sql
SELECT CONCAT(e.first_name, ' ', e.last_name) AS Employee_name,
       j.job_title,
       h.*
FROM employees e
JOIN
  (SELECT MAX(start_date),
          MAX(end_date),
          employee_id
   FROM job_history
   GROUP BY employee_id) h ON e.employee_id=h.employee_id
JOIN jobs j ON j.job_id=e.job_id
WHERE e.commission_pct = 0;
```

# *[26. Write a query in SQL to display the department name, department ID, and number of employees in each of the department.](https://www.w3resource.com/sql-exercises/joins-hr/sql-joins-hr-exercise-26.php)*

```sql
SELECT d.department_name,
       e.*
FROM departments d
JOIN
  (SELECT count(employee_id),
          department_id
   FROM employees
   GROUP BY department_id) e USING (department_id);
```

# 子查询的意义

创建满足查询条件的临时视图view

子查询就是将用来定义视图的SELECT语句直接用于FROM子句当中。先执行FROM中的SELECT语句再执行外面的语句。由内而外，跟嵌套函数是一样的，跟矩阵变换是一样的。这是因为SQL语句的执行顺序所致。

子查询不可以使用ORDER BY

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

子查询与join都是多表查询，他们的区别在哪里？

子查询用于多重条件？什么的什么的数据，子查询查询的是第一个什么的。

子查询用于展示的内容来源于一个table，而展示条件却需要用到其他table的情况。子查询生成的view。

join用于展示的内容来源于多个table。

FROM没有子查询

# [3. Write a query to find all the orders issued against the salesman who works for customer whose id is 3007.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-3.php)

```sql
-- 找出发出客户ID为3007的订单的销售人员
-- distinct 取唯一的值，因为它是子查询目的就是输出值，而不是为了查看
--
SELECT *
FROM orders
WHERE salesman_id =
    (SELECT DISTINCT salesman_id 
     FROM orders 
     WHERE customer_id =3007);

```



# [8. Write a query to count the customers with grades above New York's average.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-8.php)

```sql
SELECT grade, COUNT (DISTINCT customer_id)
FROM customer
GROUP BY grade
HAVING grade >
    (SELECT AVG(grade)
     FROM customer
     WHERE city = 'New York');

```

# 子查询与JOIN的转换

[9 Write a query to display all customers with orders on October 5, 2012.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-9.php)

```sql
--子查询
SELECT *
FROM customer 
WHERE customer_id IN 
(SELECT customer_id
 FROM orders
 WHERE ord_date = '2012-10-05');
 
--连接查询
SELECT C.*
FROM customer C
INNER JOIN orers O
ON C.customer_id = O.customer_id
AND O.ord_date = '2012-10-05';
```

子查询和链接查询，需要根据查询的列名来看怎么使用。如果要呈现列名的列名在不同的表中，那么就必须使用JOIN。如果要呈现的列名在同一个表中，那么使用子查询或JOIN都可以。

# [11. Write a query to find the name and numbers of all salesmen who had more than one customer.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-11.php)

```sql
SELECT salesman_id,name 
FROM salesman a 
WHERE 1 < 
    (SELECT COUNT(*) 
     FROM customer 
     WHERE salesman_id=a.salesman_id);
```



# [14.Write a query to find the sums of the amounts from the orders table, grouped by date, eliminating all those dates where the sum was not at least 1000.00 above the maximum amount for that date.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-14.php)

```sql
SELECT ord_date, SUM (purch_amt)
FROM orders a
GROUP BY ord_date
HAVING SUM (purch_amt) >
    (SELECT 1000.00 + MAX(purch_amt) 
     FROM orders b 
     WHERE a.ord_date = b.ord_date);

```

# [15. Write a query to extract the data from the customer table if and only if one or more of the customers in the customer table are located in London.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-15.php)完全读不懂

```sql
SELECT customer_id,cust_name, city
FROM customer
WHERE EXISTS
   (SELECT *
    FROM customer 
    WHERE city='London');
```

[EXISTS 和 IN的区别](https://www.jianshu.com/p/f212527d76ff)

https://blog.csdn.net/zhangsify/article/details/71937745

exists 和 IN的功能一样，但是效率较高。

```sql
select * from A where id in (select id from B);

select * from A where exists (select id from B where A.id=B.id);
```

EXISTS用于检查子查询是否至少会返回一行数据，该子查询实际上并不返回任何数据，而是返回值True或False。那么，这里我们来看一下in和exists的区别：
①in 是把外表和内表作hash 连接，而exists是对外表作loop循环，每次loop循环再对内表进行查询。
②一直以来认为exists比in效率高的说法是不准确的。
　-->如果查询的两个表大小相当，那么用in和exists差别不大。
　-->***如果两个表中一个较小，一个是大表，则子查询表大的用exists，子查询表小的用in***

# [16. Write a query to find the salesmen who have multiple customers.](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-16.php)完全不懂

```sql
SELECT * 
FROM salesman 
WHERE salesman_id IN (
   SELECT DISTINCT salesman_id 
   FROM customer a 
   WHERE EXISTS (
      SELECT * 
      FROM customer b 
      WHERE b.salesman_id=a.salesman_id 
      AND b.cust_name<>a.cust_name));

SELECT a.salesman_id, a.name, count(*)
FROM salesman a, customer b
WHERE a.salesman_id = b.salesman_id
GROUP BY 1, 2
HAVING COUNT(*) >= 2;
```



# 子查询中 EXISTS IN ANY ALL的用法

https://blog.csdn.net/tjuyanming/article/details/77015427

https://blog.csdn.net/zzxian/article/details/7278682?utm_source=blogxgwz1

https://www.cnblogs.com/feiquan/p/8654171.html 讲的特别清楚

为了实现大于一群值，小于一群值而出现的。相当判断一个值 对一个集合的关系。

```sql
--EXISTS
SELECT column_name(s)
FROM table_name
WHERE EXISTS
(SELECT column_name FROM table_name WHERE condition)

--IN
SELECT column_name(s)
FROM table_name
WHERE column_name IN (SELECT STATEMENT)

--ANY
SELECT column_name(s)
FROM table_name
WHERE column_name operator ANY
(SELECT column_name FROM table_name WHERE condition)

--ALL
SELECT column_name(s)
FROM table_name
WHERE column_name operator ALL
(SELECT column_name FROM table_name WHERE condition)

```

in 和 =any 即满足一个即可

not in 和  <> all 即不等于所有，每个都不相等。

但是<>any 只要有一个不相等即为true



# [19.Write a query to find salesmen with all information who lives in the city where any of the customers lives](https://www.w3resource.com/sql-exercises/subqueries/sql-subqueries-inventory-exercise-19.php)

```sql
--EXISTS
SELECT *
FROM salesman S
WHERE EXISTS (
SELECT city FROM customer C
WHERE S.city = C.city );

--IN
SELECT *
FROM salesman
WHERE city IN
(SELECT city
 FROM customer
 JOIN salesman USING(city));
 
 --ANY
 SELECT *
FROM salesman 
WHERE city=ANY
    (SELECT city
     FROM customer);
```



# FROM 后面的子查询

子查询的例子

https://www.cnblogs.com/wangshenhe/archive/2012/11/28/2792093.html



# [Write a query to display all the information of an employee whose salary and reporting person id is 3000 and 121 respectively.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-7.php)

分别符合某些条件

```sql
SELECT * 
FROM employees 
WHERE (salary,manager_id)=
(SELECT 3000,121);
```



# [31.Write a query which is looking for the names of all employees whose salary is greater than 50% of their department’s total salary bill.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-31.php)

```sql
--我的错误解答
SELECT first_name, last_name
FROM employees
WHERE salary > (
SELECT (SUM(salary))*0.5
FROM employees
GROUP BY department_id);

--正确答案
SELECT e1.first_name, e1.last_name 
FROM employees e1 
WHERE salary > 
( SELECT (SUM(salary))*.5 
FROM employees e2 
WHERE e1.department_id=e2.department_id);
```

要按照department_id 来做对比

# [Write a query to display the employee id, name ( first name and last name ), salary, department name and city for all the employees who gets the salary as the salary earn by the employee which is maximum within the joining person January 1st, 2002 and December 31st, 2003.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-34.php)

在使用join的时候还是要用where 来添加筛选条件

```sql
SELECT E.employee_id, CONCAT(E.first_name, ' ', E.last_name), E.salary, 
D.department_name, 
L.city
FROM employees E
JOIN departments D USING(department_id)
JOIN locations L  USING(location_id)
WHERE E.salary = (
SELECT MAX(salary)
FROM employees
WHERE hire_date BETWEEN '2002-01-01' AND '2003-12-31');
```



# [游标](http://www.cnblogs.com/yangyang8848/archive/2009/07/02/1514593.html)

 游标（Cursor）是处理数据的一种方法，为了查看或者处理结果集中的数据，游标提供了在结果集中一次以行或者多行前进或向后浏览数据的能力。我们可以把游标当作一个指针，它可以指定结果中的任何位置，然后允许用户对指定位置的数据进行处理

结果集：指的是SELECT 查询到结果

**游标相当于鼠标的滑轮，可以让你在SELECT的结果中逐行产看结果。**

游标的生命周期包含有五个阶段：声明游标、打开游标、读取游标数据、关闭游标、释放游标

```sql
--声明游标
标准游标：
Declare MyCursor Cursor 
    For Select * From Master_Goods
只读游标
Declare MyCusror Cursor
    For Select * From Master_Goods
    For Read Only
可更新游标
Declare MyCusror Cursor
    For Select * From Master_Goods
    For UpDate

--打开游标
全局游标：Open Global MyCursor            
局部游标: Open MyCursor

--读取游标数据
Fetch [Next | Prior | First | Last | Absolute　n　| Relative　n　]  From MyCursor
'''
Next表示返回结果集中当前行的下一行记录，如果第一次读取则返回第一行。默认的读取选项为Next
Prior表示返回结果集中当前行的前一行记录，如果第一次读取则没有行返回，并且把游标置于第一行之前。
First表示返回结果集中的第一行，并且将其作为当前行。
Last表示返回结果集中的最后一行，并且将其作为当前行。
Absolute　n　如果n为正数，则返回从游标头开始的第n行，并且返回行变成新的当前行。如果n为负，则返回从游标末尾开始的第n行，并且返回行为新的当前行，如果n为0，则返回当前行。
Relative　n　如果n为正数，则返回从当前行开始的第n行，如果n为负,则返回从当前行之前的第n行，如果为0，则返回当前行。
'''

--关闭游标
Close Global MyCursor               Close MyCursor

--释放游标
Deallocate Glboal MyCursor       Deallocate MyCursor

--实例
Declare MyCusror Cursor Scroll
    For Select * From Master_Goods Order By GoodsID
Open MyCursor
    Fetch next From MyCursor
    Into @GoodsCode,@GoodsName
    While(@@Fetch_Status = 0)
        Begin
            Begin
                Select @GoodsCode = Convert(Char(20),@GoodsCode)
                Select @GoodsName = Convert(Char(20),@GoodsName)
                PRINT @GoodsCode + ':' + @GoodsName
                End
                Fetch next From MyCursor
                Into @GoodsCode,@GoodsName
                End
Close MyCursor
Deallocate MyCursor
```



目前不懂的地方是 declare begin-end 这两个地方。



# [48. Write a query in SQL to display the the details of those departments which max salary is 7000 or above for those employees who already done one or more jobs.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-48.php)

```sql
SELECT *
FROM departments
WHERE DEPARTMENT_ID IN
    (SELECT DEPARTMENT_ID
     FROM employees
     WHERE EMPLOYEE_ID IN
         (SELECT EMPLOYEE_ID
          FROM job_history
          GROUP BY EMPLOYEE_ID
          HAVING COUNT(EMPLOYEE_ID) > 1)
     GROUP BY DEPARTMENT_ID
     HAVING MAX(SALARY) > 7000);
```

# [52.Write a query in SQL to display all the infromation about those employees who earn second lowest salary of all the employees.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-52.php)

```sql
--我的答案
SELECT * 
FROM employees
WHERE salary = (
SELECT MIN(salary)
FROM employees
WHERE salary >(
SELECT MIN(salary)
FROM employees));

--官方答案
SELECT *
FROM employees m
WHERE  2 = (SELECT COUNT(DISTINCT salary ) 
            FROM employees
            WHERE  salary <= m.salary);
```



# [54. Write a query in SQL to display the department ID, full name (first and last name), salary for those employees who is highest salary drawar in a department.](https://www.w3resource.com/sql-exercises/sql-subqueries-exercise-54.php)

```sql
SELECT department_id, first_name || ' ' || last_name AS Employee_name, salary 
	FROM employees a
		WHERE salary = 
			(SELECT MAX(salary) 
				FROM employees 
			WHERE department_id = a.department_id);
```

子查询：选取对应的部门的最大值。



# [UNION](http://www.runoob.com/sql/sql-union.html)

https://blog.csdn.net/zouxucong/article/details/73468979

UNION 操作符用于合并两个或多个 SELECT 语句的结果集。

请注意，UNION 内部的 SELECT 语句必须拥有相同数量的列。列也必须拥有相似的数据类型。同时，每条 SELECT 语句中的列的顺序必须相同。

union操作符合并的结果集，不会允许重复值，如果允许有重复值的话，使用UNION ALL.

所谓的重复值是指这条记录完全一样

UNION结果集中的列名总等于union中第一个select语句中的列名

```sql
SELECT salesman_id "ID", name, 'Salesman'
FROM salesman
WHERE city='London'
UNION
(SELECT customer_id "ID", cust_name, 'Customer'
FROM customer
WHERE city='London')
```

这里注意：由于列名的使用了第一个select的列名，所以在结果中要注意区分数据来源。



# [4. Write a query to make a report of which salesman produce the largest and smallest orders on each date.](https://www.w3resource.com/sql-exercises/union/sql-union-exercise-4.php)

```sql
SELECT a.salesman_id, name, ord_no, 'highest on', ord_date
FROM salesman a, orders b
WHERE a.salesman_id =b.salesman_id
AND b.purch_amt=
	(SELECT MAX (purch_amt)
	FROM orders c
	WHERE c.ord_date = b.ord_date)
UNION
(SELECT a.salesman_id, name, ord_no, 'lowest on', ord_date
FROM salesman a, orders b
WHERE a.salesman_id =b.salesman_id
AND b.purch_amt=
	(SELECT MIN (purch_amt)
	FROM orders c
	WHERE c.ord_date = b.ord_date))

```

UNION相当于将两个查询结果拼接在一起了。

# [6. Write a query to list all the salesmen, and indicate those who do not have customers in their cities, as well as whose who do.](https://www.w3resource.com/sql-exercises/union/sql-union-exercise-6.php)

```sql
SELECT salesman.salesman_id, name, cust_name, commission
FROM salesman, customer
WHERE salesman.city = customer.city
UNION
(SELECT salesman_id, name, 'NO MATCH', commission
FROM salesman
WHERE NOT city = ANY
	(SELECT city
        FROM customer))
ORDER BY 2 DESC

```



# MySQL导入导出CSV

```sql
# 导出CSV
SELECT * FROM [TABLE]
INTO OUTFILE '[FILE]'；
或者
SELECT * FROM [TABLE]
INTO OUTFILE '[FILE]'
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' 
LINES TERMINATED BY '\n'；
# 导入CSV
LOAD DATA INFILE '[FILE]'
INTO TABLE [TABLE]；
或者
LOAD DATA INFILE '[FILE]'
INTO TABLE [TABLE]
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' 
LINES TERMINATED BY '\n'；
```

# [事务](http://www.runoob.com/mysql/mysql-transaction.html)

# [新的练习](https://zhuanlan.zhihu.com/p/38354000)

# case when
```sql
select c_id, (及格人数 / 总人数)*100 as '及格百分数'
from 
(select c_id ,
sum(case when s_score>=60 then 1 else 0 end) as '及格人数',
sum(case when s_score<60 then 1 else 0 end) as '不及格人数',
(select count(s_score) from Score) as '总人数' 
from score
group by c_id ) as per
```

# 查询和“01”号同学所学课程完全相同的其他同学的学号
【解题思路】首先找出学号为1001的学生学习过的课程，然后根据这些课程号和所学课程总数就可以找到有哪些同学学习过和他一样的课程
```sql
select s_id
from Score
where c_id in
(select c_id from Score where s_id='01')
and s_id <> '01'
group by s_id
having count(c_id)=(select count(c_id) from Score where s_id='01');
```

# 把“SCORE”表中“张三”老师教的课的成绩都更改为此课程的平均成绩
【解题思路】考察数据库更改操作。首先找到李多多老师教过哪些课程及其课程的平均成绩，然后根据课程号关联成绩表进行更新
```sql
update Score as a join 
(select avg(s_score) as t, Score.c_id from Score 
join Course on Score.c_id= Course.c_id
join Teacher on Teacher.t_id= Course.t_id
where t_name ='张三' group by c_id) as b#张三老师教的课与平均分
on a.c_id= b.c_id
set a.s_score= b.t;
```

# ***按各科平均成绩从低到高和及格率的百分数从高到低排列，以如下形式显示：***
子查询建好表，然后从表中提取数据
```sql
select partOne.c_id as '课程号',课程名,平均成绩,及格百分数
FROM
(select c.c_id, c.c_name as '课程名', avg(s.s_score) as '平均成绩'
from Course c join Score s
on c.c_id = s.c_id
group by c.c_id) as partOne 
join 
(select c_id, (及格人数 / 该课程总人数)*100 as '及格百分数'
from 
(select c_id ,
sum(case when s_score>=60 then 1 else 0 end) as '及格人数',
sum(case when s_score<60 then 1 else 0 end) as '不及格人数',
count(*) as '该课程总人数' 
from Score
group by c_id ) as partTwo) as partThree
on partOne.c_id = partThree.c_id;
```

# ***查询学生平均成绩及其名次***

```sql
SELECT s_id as '学号',平均成绩,
      (SELECT COUNT(*) FROM(SELECT s_id,AVG(s_score)AS '平均成绩'
                            FROM Score GROUP BY s_id)AS b 
       WHERE b.平均成绩>a.平均成绩)+1 as RANK
FROM (select s_id,avg(S_score) as 平均成绩 from Score group by s_id)AS a
order by 平均成绩 desc;
```
[讲解](https://blog.csdn.net/djlzxzy/article/details/3897069)

# ***查询每门课程成绩最好的前两名***
```sql
SELECT a.c_id as 课程号,c_name as 课程名称,s_name as 姓名,b.s_score as chengji  FROM 
Course a join Score b on a.c_id=b.c_id
join Student c on c.s_id=b.s_id
WHERE (SELECT COUNT(*) FROM Score d WHERE a.c_id=d.c_id AND b.s_score<d.s_score)<=1
ORDER BY a.c_id ASC,b.s_score DESC
```

# 查询学过编号为“01”的课程并且也学过编号为“02”的课程的学生的学号、姓名
```sql
--写法一：找到合适的s_id,从student中选取数据
select s_id,s_name
from Student 
where s_id in
(select s_id from Score where c_id = '01') 
AND s_id in
(select s_id from Score where c_id = '02')
--写法二：制作合适的临时表，从中提取满足需求的数据，
select a.s_id,a.s_name
from Student a JOIN Score b ON a.s_id=b.s_id
JOIN Score c ON a.s_id=c.s_id
where b.c_id='01' and c.c_id='02'
--确保是同一个学生的选课数据
and b.s_id=c.s_id ;
```

# 42、查询有2门不同课程成绩相同的学生的学号、课程号、学生成绩
多条件join
```sql
select distinct a.s_id as 学生编号 ,a.c_id as 课程编号,a.s_score as 学生成绩
from Score a join Score b
on a.s_id=b.s_id and a.c_id<> b.c_id
where a.s_score=b.s_score;
```

# [leetcode 第n高的薪资](https://leetcode.com/problems/nth-highest-salary/)
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
Declare M int;
Set M = N-1;
  RETURN (
      # Write your MySQL query statement below.
      select Distinct salary from Employee order by Salary limit M, 1 
  );
END
```
## [limit offset](https://blog.csdn.net/iastro/article/details/53037600)
这个功能类似pandas的切片 limit 限制返回多少行， offset 告知从多少行开始返回
```sql
limit 9,4
等价于
Limit 4 offset 9
均返回 10,11,12,13
```
## [自定义函数](https://www.cnblogs.com/caoruiy/p/4485273.html)

# [leetcode 第二高的薪资](https://leetcode.com/problems/second-highest-salary/solution/)
```sql
SELECT
    IFNULL(
      (SELECT DISTINCT Salary
       FROM Employee
       ORDER BY Salary DESC
        LIMIT 1 OFFSET 1),
    NULL) AS SecondHighestSalary
```
## ifnull
https://www.cnblogs.com/acm-bingzi/p/mysqlIfnull.html

# [leetcode rank 成绩](https://leetcode.com/problems/rank-scores/)
```sql
--详细解释
select @start:=0; -- 自定义变量且返回一个表，只有一行且值为0
select @start:=@start+1 from (select @start:=0) b;-- 对自定变量进行运算，返回运算结果
select distinct b.*, a.* from 178_RankScores a, (select @start:=0) b;-- 将自定义变量的表与其他表拼接，类似pd.concat
select @start:=@start+1, a.* from 178_RankScores a, (select @start:=0) b;-- 对自定义变量进行运算，再拼接
select @start:=@start+1, a.Score from 178_RankScores a, (select @start:=0) b;--同上
select  distinct @start:=@start+1, a.Score from 178_RankScores a, (select @start:=0) b; --因为自定变量这里的结果可以理解为行号，所以是唯一值
select Score from 178_RankScores order by Score desc;
select distinct Score from 178_RankScores order by Score desc;


--解法1 更快 创建包含要提取的字段表，并从中提取
select X.Score, Rank 
from Scores, (
    select Score,(@start:=@start+1) as Rank --创建的第一个表，包含Score和rank，对score取唯一值并计算行号也就是rank
    from (
        select distinct Score from Scores order by Score desc) as S, 
        (select @start := 0) as T) as X --创建第二个表 获得rank，
where X.Score=Scores.Score 
order by Rank;
--解法2 
 SELECT S.Score, c.Rank
FROM Scores as S LEFT OUTER JOIN 
(SELECT b.Score, 
  (SELECT count(*) 
     FROM (select distinct Score from Scores) as a 
     WHERE b.Score <= a.Score
  ) AS Rank
FROM (select distinct Score from Scores) as b   
ORDER BY Score DESC
 ) AS c
ON S.Score = c.Score
ORDER BY Score DESC;


```
https://www.cnblogs.com/genialx/p/5932558.html
@ 变量 声明自定义变量
https://www.cnblogs.com/genialx/p/5932558.html
变量的使用
https://blog.csdn.net/lili625/article/details/80252420
行号的使用

:=的作用 为变量赋值， = 的作用是用于比较的
## 四种rank的方法
With Variables: 841 ms

First one uses two variables, one for the current rank and one for the previous score.

SELECT
  Score,
  @rank := @rank + (@prev <> (@prev := Score)) Rank
FROM
  Scores,
  (SELECT @rank := 0, @prev := -1) init
ORDER BY Score desc
Always Count: 1322 ms

This one counts, for each score, the number of distinct greater or equal scores.

SELECT
  Score,
  (SELECT count(distinct Score) FROM Scores WHERE Score >= s.Score) Rank
FROM Scores s
ORDER BY Score desc
Always Count, Pre-uniqued: 795 ms

Same as the previous one, but faster because I have a subquery that "uniquifies" the scores first. Not entirely sure why it's faster, I'm guessing MySQL makes tmp a temporary table and uses it for every outer Score.

SELECT
  Score,
  (SELECT count(*) FROM (SELECT distinct Score s FROM Scores) tmp WHERE s >= Score) Rank
FROM Scores
ORDER BY Score desc
Filter/count Scores^2: 1414 ms

Inspired by the attempt in wangkan2001's answer. Finally Id is good for something :-)

SELECT s.Score, count(distinct t.score) Rank
FROM Scores s JOIN Scores t ON s.Score <= t.score
GROUP BY s.Id
ORDER BY s.Score desc

# [distinct 用法](https://blog.csdn.net/lmy86263/article/details/73612020)
```sql
SELECT country, distinct province from person; // 该语句是错误的
```

# sql 优化
# sql join
https://blog.csdn.net/forlovehuan/article/details/78979831
所有的表链接都是在笛卡尔积的基础 进行选取的。遍历多个表的笛卡尔积，从中选择符合where的数据，做成表。
笛卡尔积的结果是 左表行数×右表行数， on 限制了在那个字段上进行笛卡尔积。只有字段相同的值才会进行笛卡尔积（on 和 where的作用）

# [group by 的分组操作](https://blog.csdn.net/u013166209/article/details/52605926)
在查询的时候，当使用group by之后，每个分组默认显示的数据是主键值最小的那一条数据，这个时候使用再使用 order by 来排序的话也也只是对分组后的结果进行排序，对分组内数据的排序没有影响。
记住运行顺序，先group by 再 Order by。order by ， limit都是对临时表的操作

# [max min group by的坑](https://blog.csdn.net/hj12312/article/details/79376215)
[max min 的作用范围](http://www.cnblogs.com/kuangwong/p/8027733.html)
max min 只作用于一列，Group by 只返回该组中出现的第一个值
当要进行分组最大值，最小值的时候，要使用子查询和where

# [truncate table](https://www.1keydata.com/cn/sql/sql-truncate.php)
清除表的内容，而不删除（drop）表。

# [185. Department Top Three Salaries](https://leetcode.com/problems/department-top-three-salaries/)
疑问：
1、以下两个sql，结果为什么不同。
**2、问题根源是子查询中的字段**
```sql
# 正确答案
select e.Name, d.Name, e.Salary
from 185_Employee e left join 185_Department d
on e.DepartmentId=d.Id
where 
(select count(distinct e.Salary)
from 185_Employee e2 --这里没有联结
where e.DepartmentId = e2.DepartmentId
and e2.Salary>e.Salary)<=3

# 结果为空
select e.Name, d.Name, e.Salary
from 185_Employee e left join 185_Department d
on e.DepartmentId=d.Id
where 
(select count(distinct e1.Salary)
from 185_Employee e1 join 185_Employee e2  --这里有一个链接
on e1.DepartmentId = e2.DepartmentId
where e2.Salary>e1.Salary)<=3

# 将子查询的e1改成e 结果正确
select e.Name, d.Name, e.Salary
from 185_Employee e left join 185_Department d
on e.DepartmentId=d.Id
where 
(select count(distinct e.Salary) --这里改成e.Salary
from 185_Employee e1 join 185_Employee e2  
on e1.DepartmentId = e2.DepartmentId
where e2.Salary>e1.Salary)<=3
```

# [sql delete 联结删除](https://blog.csdn.net/yun__yang/article/details/53485760)
```sql
delete table1 from table1 inner join table2 on table1.id = table2.id where table2.type = 'something' and table1.id = 'idnums';
```

# [Mysql 日期](http://www.runoob.com/sql/sql-dates.html)
```sql
NOW()	返回当前的日期和时间
CURDATE()	返回当前的日期
CURTIME()	返回当前的时间
DATE()	提取日期或日期/时间表达式的日期部分
EXTRACT()	返回日期/时间的单独部分
DATE_ADD()	向日期添加指定的时间间隔
DATE_SUB()	从日期减去指定的时间间隔
DATEDIFF()	返回两个日期之间的天数
DATE_FORMAT()	用不同的格式显示日期/时间
```

# [mysql update](http://www.runoob.com/mysql/mysql-update-query.html)
```sql
UPDATE table_name SET field1=new-value1, field2=new-value2
[WHERE Clause]
```

# [MySQL 数据区间分组](https://www.cnblogs.com/lazyx/p/5577105.html)
https://www.cnblogs.com/shea/archive/2017/04/25/6760381.html
elt , interval
ELT(n,str1,str2,str3,...) ：如果n=1，则返回str1,如果n=2，则返回str2,依次类推。如果n小于1或大于参数个数，返回NULL
INTERVAL()函数进行比较列表(N，N1，N2，N3等等)中的N值。该函数如果N<N1返回0，如果N<N2返回1，如果N<N3返回2 等等。如果N为NULL，它将返回-1

# [MySQL 行列转化](https://www.cnblogs.com/angryprogrammer/p/5249472.html)
思想：与pivot一样，需要手动制作表，行名通过group by来实现，列名通过if或case when实现， 值通过对列名的操作来实现（如聚合）
IF(expr1,expr2,expr3)，如果expr1的值为true，则返回expr2的值，如果expr1的值为false，
IFNULL(expr1, expr2), 如果expr1不是NULL，IFNULL()返回expr1，否则它返回expr2。IFNULL()返回一个数字或字符串值
```sql
--case when 与 if的转换
select 
ifnull(c1, 'total'),
sum(case when c2='B1' then c3 else 0 end) as B1,
sum(case c2 when 'B2' then c3 else 0 end) as B2, 
sum(if(c2='B3', c3, 0)) as B3,
sum(if(c2='B4', c3, 0)) as B4,
sum(c3) as total
from tx
group by c1 with rollup;
```