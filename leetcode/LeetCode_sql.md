# @符号的意义

在SQL Server中以这些符号作为标识符的开始具有特殊的含义。一个以at符号(@)开头的标识符表示一个本地的变量或者参数。一个以数字符号(#)开头的标识符代表一个临时表或者过程。一个以两个数字符号(##)开头的标识符标识的是一个全局临时对象

# 别名 alias

as 为列设定别名

SELECT × as y from a as b，将输出的表名字设定为b, 列名x设定成y