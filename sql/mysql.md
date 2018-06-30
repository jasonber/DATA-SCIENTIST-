# 语句执行顺序

输入顺序

1、 SELECT   2、 FROM  3、WHERE  4、GROUP BY 5、 HAVING 6、ORDER BY

执行顺序

1、FROM  2、 WHERE 3、 GROUP BY  4、HAVING 5、 SELECT 6、 ORDER BY

# WHERE 和 having的区别

Where 是一个约束声明，使用Where约束来自数据库的数据，Where是在结果返回之前起作用的，Where中不能使用聚合函数。  Having是一个过滤声明，是在查询返回结果集以后对查询结果进行的过滤操作，在Having中可以使用聚合函数。  在查询过程中聚合语句(sum,min,max,avg,count)要比having子句优先执行。而where子句在查询过程中执行优先级高于聚合语句。

WHERE = 指定行所对应的条件

HAVING = 指定组所对应的条件