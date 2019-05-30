create table  COURSE
(`CNO`      string  comment '课程编号',
`CNAME`     string   comment '课程名称,',
`TNO`       string     comment '教师编号'
) row format delimited fields terminated by '\t' lines terminated by '\n',
--课程表数据插入
insert into  COURSE  values   ('3-105' ,'计算机导论',825)
                                        ,('3-245' ,'操作系统' ,804)
                                        ,('6-166' ,'数据电路' ,856)
                                        ,('9-888' ,'高等数学' ,100),
                                        
show tables,
select * from course,

create table  SCORE
(`SNO`      string  comment '学生编号',
`CNO`       string  comment '课程编号',
`DEGREE`    int  comment '分数'
),
--成绩表数据插入
insert into  SCORE  values   (103,'3-245',86)
                                        ,(105,'3-245',75)
                                        ,(109,'3-245',68)
                                        ,(103,'3-105',92)
                                        ,(105,'3-105',88)
                                        ,(109,'3-105',76)
                                        ,(101,'3-105',64)
                                        ,(107,'3-105',91)
                                        ,(108,'3-105',78)
                                        ,(101,'6-166',85)
                                        ,(107,'6-106',79)
                                        ,(108,'6-166',81),

                                      
create table  Student
(`SNO`      string  comment '学生编号',
`SNAME`     string  comment '学生姓名',
`SSEX`      string     comment '学生性别',
`SBIRTHDAY` string     comment '出生年月',
`CLASS` string     comment '班级'
)
row format delimited fields terminated by '\t' lines terminated by '\n',
--学生表数据插入
insert into  Student  values  (108 ,'曾华' ,'男' ,1977-09-01,95033)
                                        ,(105 ,'匡明' ,'男' ,1975-10-02,95031)
                                        ,(107 ,'王丽' ,'女' ,1976-01-23,95033)
                                        ,(101 ,'李军' ,'男' ,1976-02-20,95033)
                                        ,(109 ,'王芳' ,'女' ,1975-02-10,95031)
                                        ,(103 ,'陆君' ,'男' ,1974-06-03,95031),
                                        
select * from student,`


create table  TEACHER
(`TNO`      string  comment '教师编号',
`TNAME`     string  comment '教师姓名',
`TSEX`      string     comment '性别',
`TBIRTHDAY` string     comment '出生年月',
`PROF` string     comment '职称',
`DEPART` string     comment '课程'
),


insert into  TEACHER  values  (804,'李诚','男','1958-12-02','副教授','计算机系')
                                        ,(856,'张旭','男','1969-03-12','讲师','电子工程系')
                                        ,(825,'王萍','女','1972-05-05','助教','计算机系')
                                        , (831,'刘冰','女','1977-08-14','助教','电子工程系'),
                                        
select * from teacher,

set mapreduce.map.memroy.mb,

select * from course,
select * from score,
select * from student,
select * from teacher,
drop table student,

select * from score
where degree >60 and degree<80,

select * from score
where degree in (85, 86, 88),
drop table course,
drop table score,
drop table teacher,
drop table student,

-- 学生表
CREATE TABLE Student(
s_id VARCHAR(20),
s_name VARCHAR(20) ,
s_birth VARCHAR(20) ,
s_sex VARCHAR(10) 
),

-- 课程表
CREATE TABLE Course(
c_id VARCHAR(20),
c_name VARCHAR(20),
t_id VARCHAR(20)
),

-- 教师表
CREATE TABLE Teacher(
t_id VARCHAR(20),
t_name VARCHAR(20)
),

-- 成绩表
CREATE TABLE `Score`(
s_id VARCHAR(20),
c_id VARCHAR(20),
s_score INT
),


-- 插入学生表测试数据
insert into student values ('01' , '赵雷' , '1990-01-01' , '男'),
('02' , '钱电' , '1990-12-21' , '男'),
('03' , '孙风' , '1990-05-20' , '男'),
('04' , '李云' , '1990-08-06' , '男'),
('05' , '周梅' , '1991-12-01' , '女'),
('06' , '吴兰' , '1992-03-01' , '女'),
('07' , '郑竹' , '1989-07-01' , '女'),
('08' , '王菊' , '1990-01-20' , '女');

-- 课程表测试数据
insert into course values ('01' , '语文' , '02'),
('02' , '数学' , '01'),
('03' , '英语' , '03');

-- 教师表测试数据
insert into Teacher values('01' , '张三'),
('02' , '李四'),
('03' , '王五');

-- 成绩表测试数据
insert into Score values ('01' , '01' , 80),
('01' , '02' , 90),
('01' , '03' , 99),
('02' , '01' , 70),
('02' , '02' , 60),
('02' , '03' , 80),
('03' , '01' , 80),
('03' , '02' , 80),
('03' , '03' , 80),
('04' , '01' , 50),
('04' , '02' , 30),
('04' , '03' , 20),
('05' , '01' , 76),
('05' , '02' , 87),
('06' , '01' , 31),
('06' , '03' , 34),
('07' , '02' , 89),
('07' , '03' , 98);

select * from course;
select * from score;
select * from student;
select * from teacher;

select count(t_id) from teacher
where t_name like '张%';

drop table course;
drop table score;
drop table student;
drop table teacher;

#创建表
CREATE TABLE Course (`c_id` varchar(20), `c_name` varchar(20),   `t_id` varchar(20)) row format delimited fields terminated by ','; 
CREATE TABLE `Score` (   `s_id` varchar(20) ,   `c_id` varchar(20) ,   `s_score` int) row format delimited fields terminated by ','; 
CREATE TABLE `Student` (   `s_id` varchar(20) ,   `s_name` varchar(20),   `s_birth` varchar(20),   `s_sex` varchar(10)) row format delimited fields terminated by ','; 
CREATE TABLE `Teacher` (   `t_id` varchar(20) ,   `t_name` varchar(20)) row format delimited fields terminated by ','; 
CREATE TABLE `178_RankScores` (   `Id` int NOT NULL,   `Score` float ) row format delimited fields terminated by ','; 
CREATE TABLE `180_consecutive` (   `Id` int NOT NULL,   `Num` int NOT NULL ) row format delimited fields terminated by ','; 
CREATE TABLE `184_Department` (   `Id` int,   `Name` varchar(20)  ) row format delimited fields terminated by ',';
CREATE TABLE `184_Employee` (   `Id` int,   `Name` varchar(20) ,   `Salary` int,   `DepartmentId` varchar(20)  ) row format delimited fields terminated by ','; 
CREATE TABLE `185_Department` (   `Id` int ,   `Name` varchar(20)  ) row format delimited fields terminated by ','; 
CREATE TABLE `185_Employee` (   `Id` int ,   `Name` varchar(20) ,   `Salary` int ,   `DepartmentId` varchar(20)  ) row format delimited fields terminated by ',';
CREATE TABLE `bins` (   `id` int ,   `yb` int  ) row format delimited fields terminated by ',';
CREATE TABLE `country_pop` (   `country` varchar(255) ,   `sex` varchar(255) ,   `population` int  ) row format delimited fields terminated by ','; 
CREATE TABLE `course2` (   `s_id` varchar(255) ,   `c_id` varchar(255) ,   `course` varchar(255) ,   `flag` varchar(255)  ) row format delimited fields terminated by ','; 
CREATE TABLE `tx` (   `id` int NOT NULL,   `c1` char(2) ,   `c2` char(2) ,   `c3` int    ) row format delimited fields terminated by ','; 
CREATE TABLE `company_mast` (   `COM_ID` bigint ,   `COM_NAME` varchar(255)  ) row format delimited fields terminated by ','; 
CREATE TABLE `customer` (   `customer_id` bigint ,   `cust_name` varchar(255) ,   `city` varchar(255) ,   `grade` double ,   `salesman_id` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `departments` (   `DEPARTMENT_ID` bigint ,   `DEPARTMENT_NAME` varchar(255) ,   `MANAGER_ID` bigint,   `LOCATION_ID` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `employees` (   `EMPLOYEE_ID` bigint ,   `FIRST_NAME` varchar(255) ,   `LAST_NAME` varchar(255) ,   `EMAIL` varchar(255) ,   `PHONE_NUMBER` varchar(255) ,   `HIRE_DATE` varchar(255) ,   `JOB_ID` varchar(255) ,   `SALARY` bigint ,   `COMMISSION_PCT` double ,   `MANAGER_ID` bigint ,   `DEPARTMENT_ID` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `item_mast` (   `PRO_ID` bigint ,   `PRO_NAME` varchar(255) ,   `PRO_PRICE` bigint ,   `PRO_COM` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `job_history` (   `EMPLOYEE_ID` bigint ,   `START_DATE` varchar(255) ,   `END_DATE` varchar(255) ,   `JOB_ID` varchar(255) ,   `DEPARTMENT_ID` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `job_id` (   `EMPLOYEE_ID` bigint ,   `start_date` varchar(255) ,   `end_date` varchar(255) ,   `JOB_ID` varchar(255) ,   `DEPARTMENT_ID` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `jobs` (   `JOB_ID` varchar(255) ,   `JOB_TITLE` varchar(255) ,   `MIN_SALARY` bigint ,   `MAX_SALARY` bigint  ) row format delimited fields terminated by ','; 
CREATE TABLE `orders` (   `ord_no` bigint ,   `purch_amt` double ,   `ord_date` varchar(255) ,   `customer_id` bigint ,   `salesman_id` varchar(255)  ) row format delimited fields terminated by ','; 
CREATE TABLE `salesman` (   `salesman_id` varchar(255) ,   `name` varchar(255) ,   `city` varchar(255) ,   `commission` double  ) row format delimited fields terminated by ',';
CREATE TABLE `scientist` (   `YEAR` varchar(255) ,   `SUBJECT` varchar(255) ,   `WINNER` varchar(255) ,   `COUNTRY` varchar(255) ,   `CATEGORY` varchar(255)  ) row format delimited fields terminated by ','; 

--载入数据
load data LOCAL 
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/departments.txt' 
into table departments;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/scientist24.txt'
into table scientist;
drop table scientist;
select * from scientist;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/job_id.txt'
into table job_id;
drop table job_id;
select * from job_id;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/employees.txt'
into table employees;
select * from employees;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/company_mast.txt'
into table company_mast;
select * from company_mast;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/item_mast.txt'
into table item_mast;
select * from item_mast;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/jobs.txt'
into table jobs;
select * from jobs;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/job_history.txt'
into table job_history;
select * from job_history;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/orders.txt'
into table orders;
drop table orders;
select * from orders;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/customer.txt'
into table customer;
select * from customer;

load data LOCAL
inpath '/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/sql/hive_database/salesman.txt'
into table salesman;
select * from salesman;

select * from departments;