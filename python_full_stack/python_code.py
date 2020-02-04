'''
while 循环，控制次数的循环
while 条件：
    循环体
    条件变化
当条件为假的时候跳出循环
'''

count = 1

print(count)

while count <= 8:
    print('while test ok')
    count += 1
print(count)

'''
break的使用 停止当前的循环
'''
while True:
    s = input('请输入内容:')
    if s == "q":
        break
    print('输入的内容是：'+s)
print('跳出循环')

'''
continue 停止本次循环，并跳出执行下次循环
'''
while True:
    s = input('请输入内容:')
    if s == "q":
        break
    print('输入的内容是：'+s)
    if '马化腾' in s:
        print("你输入的内容无法显示")
        continue

print('跳出循环')

'''1加到100'''

a = 1

b = 2

while b <= 100:
    a = a + b
    b += 1
print("a={0},b={1}".format(a, b))

'''
数字的逻辑运算
'''

print(0 or 1 or 5)

print(0 and 1 and 5)

print(1 and 2 and 5)

print(1 or 2 and 5)

print(0 or 2 and 5)

print(0 and 2 or 5)

print(0 and 5 or 2)

print(0 or 2 and 5)

print(2 > 3 and 1)

print(not 2 or 1 and 5)

print(0 or 4 and 3 or 7 or 9 and 6)

print(0 or 3 or 7 or 6)

print(4 and 3)

'''
while else的应用
break 彻底停止循环 不做判断
'''

a = 1

b = 2

while b <= 100:
    a = a + b
    b += 1
    # break 这里会跳出不运行 else
else:
    print('这里是else')
print("a={0},b={1}".format(a, b))

# bit_length()
a = 4
print(a.bit_length())

# ""空字符串 None 空

# 切片的负面 默认从左往右切,顾头不顾腚

s = 'alex和wusir搞基'

print(s[-2:])

print(s[-1:-3])

print(s[-3:-1])

# 替换指定位置的字符。转换成list之后，再替换，最后进行拼接
s = '我爱北京天安门，天安门前太阳升'

s_list = list(s)

s_list[8:11] = 'a'

s_str = "".join(s_list)
print(s_str)

# 长度的测量
st = '我爱北京天安门'

lst = list(st)

len(st)

len(lst)

# 字典 setdefault()
'''
setdefault()相当于执行两个操作
1. 判断原来的字典中有没有这个key，如果没有，执行新增
2. 用这个key去字典中查询，返回查到的key对应的value
'''
dic = {'及时雨':'宋江', '易大师':'剑圣'}
ret = dic.setdefault('及时雨','西门庆')
ret2 = dic.setdefault('黑旋风','李逵')

print(dic)
print(dic.get('及时雨'))
print(ret)
print(ret2)

# 结构， 解包：列表和元组可以通过多个变量同时赋值，拿到里面的值
# 同理dict中的item也可以使用这种方式来取到
a, b = (1, 2)
print(a, b)

a, b = [1, 2]
print(a, b)

a = 1, 2 # 等价于a = (1, 2)
a, b = (1, 2), (3, 4)
print(a)

dic = {'及时雨':'宋江'， '易大师':'剑圣', '薇恩':'暗影猎手'}
dic.keys()
dic.values()
dic.items()

for item in dic.items():
    k, v = item
    print(k, v)

for k, v in dic.items():
    print(k, v)

# 编码 encode 解码decode
s = "你好"
bs = s.encode("gbk") # 我们这样可以获取到GBK的⽂文字
# 把GBK转换成UTF-8
# ⾸首先要把GBK转换成unicode. 也就是需要解码
s = bs.decode("gbk") # 解码
# 然后需要进⾏行行重新编码成UTF-8
bss = s.encode("UTF-8") # 重新编码
print(bss)

# 列表清空
lst = ["周杰伦", "无间道", "霸王别姬", "宋江"]

for i in lst:  # list的循环的时候，是对index的循环，且会自增，所以不会实现clear（）的功能
    lst.remove(i)

print(lst)

del_lst = []
for i in lst:
    del_lst.append(i)

for a in del_lst:
    lst.remove(a)

print(lst)

# set增删改查

s = {'刘嘉玲', '关之琳', '王祖贤'}

s.add('郑裕玲') # 按照
print(s)

s.add('郑裕玲')
print(s)

s.update("马化腾") # 按照list的方式来更新，添加时随机的
print(s)

s.update(['张曼玉', '李若彤', '李若彤'])
print(s)

s.remove("刘嘉玲")

s.add("赵本山")
print(s)

# 深浅拷贝
# 直接赋值相当于授予了一个变量对该地址的操作权
# 拷贝是对内存的操作，并将新建的内存的操作权赋予相应的变量
# 浅拷贝 拷贝的是内存的值
lst1 = ["赵本三", "刘能", "赵四"]
lst2 = lst1.copy()
lst2 = lst1[:] # 切片产生新的内存数据
# 深拷贝 拷贝的是内存的所有内容
import copy
lst3 = copy.deepcopy(lst1)

a = [1, 2]
a[1] = a
print(a[1])

# fromkeys
dic = {'a':'123'}
s = dic.fromkeys("王健林", "思聪")
print(s)

# 文件读取
doc = open('file_control.txt', mode='r', encoding='utf-8')
content = doc.read()
doc.close()
print(content)

# 路径
# 相对路径：相对于当前目录的路径，便于拷贝
# 绝对路径: 从根目录开始的路径，url地址 
# 文件写入 file.write(content)写入时可以文件不存在，会自动新建。
# 写入操作中要加flush
# w模式。写完之后自动存入 
doc = open('file_control2.txt', mode='w', encoding='utf-8') # w模式 会覆盖原来的内容，
doc.write('写入w模式\n')
content = doc.read()
print(content)
doc.flush()
doc.close()

# w模式追加 
doc = open('file_control2.txt', mode='a', encoding='utf-8') # w模式 会清空文件，再写入
doc.write('追加\n')
# content = doc.read()
print(content)
# doc.flush() 
doc.close()
 
# b模式, 写字节模式。rb 读字节，wb写字节
# w、r、a 对文本操作，wb、rb、ab处理的是非文本
doc = open('file_control2.txt', mode='rb') 
# doc.write('追加\n')
content = doc.read()
print(content.decode('utf-8'))
# doc.flush()
doc.close()

# 读写模式r+ 默认情况下光标在文件的开头，必须先读后写
doc = open('file_control2.txt', mode='r+', encoding='utf-8') 
content = doc.read()
doc.write('追加\n')
print(content.decode('utf-8'))
doc.flush()
doc.close()

# 写读模式w+ 不用。读写操作与光标位置有关，从光标之后开始操作

# 光标，seek()。移动的是字节，移动到某个位置。
# r+写入：无操作时在开头写，有光标操作后在文件末尾写入。所有写入都是覆盖操作
 
 # 文件修改内容，读出内容，修改内容，写入新文件，删除原文件，新文件改名
 # with open('path',mode, encodding) as var_name, open('paht2', mode, encodeing) as var_name2:
 #     操作内容


 # 三元表达式：返回a的条件是 a > b，否则返回b
def get_big(a, b):
    c = a if a > b else b
    return c

print(get_big(6,9))

# 动态传参
# *var_name 位置传参，形成tuple
# **var_name 关键字传参，形成dict

# global 在局部命名空间中将局部变量声明为全局变量
# nonlocal 在局部命名空间将局部变量声明为非本局部的局部变量

# 函数名可看作变量来用，变量的操作均可用在函数名中
# 当作数据结构的元素
# 可以向其他函数传参
# 可当作返回值

# 迭代器 可迭代对象 不是一回事
# 使用dir()查看对象所具有的方法，如果有__iter__则为iterabel对象。
# 如果有__next__则为迭代器

# 生成器就是迭代器, 内置的生成器叫做迭代器，生成器是编写出来的
# 生成器跟函数一样只是把return 换成了yield

# reserved
st = '我爱北京天安门' 
r_st = reversed(st)
r_l = list(r_st)
st2 = "".join(list(r_st))
stt3 = "".join(r_l)

# bit_length() 
a = 10 
a.bit_length()

# 字符串判断
a = '123_45.6'
b = 'abc'
c = '_abc!@' 
d = '123abc'
e = '123'
f = '一壹123'
g = '1.23'
print(a.isalnum())
print(d.isalnum())
print(b.isalpha())
print(f.isnumeric())
print(g.isdigit())

# list声明
a = list('123')

# list.extend() 
lst = ["王志⽂文", "张⼀一⼭山", "苦海海⽆无涯"]
lst.extend(["麻花藤", "麻花不不疼"])
print(lst)
lst.pop(2)

# 步长修改
lst = ["太⽩白", "太⿊黑", "五⾊色", "银王", "⽇日天"]
lst[1] = "太污" # 把1号元素修改成太污
print(lst)
lst[1:4:3] = ["麻花藤", "哇靠"] # 切⽚片修改也OK. 如果步⻓长不不是1, 要注意. 元素的个
print(lst)

lst.index('银王')

# dict复习
dct = {'teacher':'mrs wang', 'student':'xiao bai', 'girl':'xiao hu'}
dct.keys()
dct.values()
dct.items()
dct.items()
# dict

li = "⻩黄花⼤大闺⼥女女"
s = "_".join(li)
print(s)

a = [1, 2]
a[1] = a
print(a[1])

a = 100
def func():
    global a # 加了了个global表示不不再局部创建这个变量量了了. ⽽而是直接使⽤用全局的a
    a = 28
    print(a)
func()
print(a)

def eat():
    print("我吃什什么啊")
    a = yield "馒头"
    print("a=",a)
    b = yield "⼤大饼"
    print("b=",b)
    c = yield "⾲韭菜盒⼦子"
    print("c=",c)
    yield "GAME OVER"
gen = eat() # 获取⽣生成器器
ret1 = gen.__next__()
print(ret1)
ret2 = gen.send("胡辣汤")
print(ret2)
ret3 = gen.send("狗粮")
print(ret3)
ret4 = gen.send("猫粮")
print(ret4)

st = "⼤大家好, 我是麻花藤"
s = slice(1, 5, 2)
print(st[s])

'''
类和方法的区别 https://www.py.cn/jishu/jichu/13133.html
1、函数要手动传self，方法不用传self。
2、如果是一个函数，用类名去调用，如果是一个方法，用对象去调用。
'''
class Foo(object):
    def __init__(self):
        self.name="haiyan"
    def func(self):
        print(self.name)

obj = Foo()
obj.func()
Foo.func(obj)

from types import FunctionType,MethodType
obj = Foo()
print(isinstance(obj.func,FunctionType))  #False
print(isinstance(obj.func,MethodType))   #True   #说明这是一个方法
print(isinstance(Foo.func,FunctionType))  #True   #说明这是一个函数。
print(isinstance(Foo.func,MethodType))  #False

# class
class Person:
    '''类体:两部分:变量部分,方法(函数)部分'''
    mind = '有思想'  # 变量,静态变量,静态字段
    animal = '高级动物'
    faith = '有信仰'

    def __init__(self,name,age,hobby):
        print(666)
        self.name = name  #  Person.money = '运用货币'
        self.age = age
        self.hobby = hobby

    def work(self):  # 方法,函数,动态变量

        print('%s都会工作...' %self.name)
    def shop(self):

        print('人类可以消费....')

ret1 = Person
print(ret1.__dict__)

ret2 = Person('test', 18, 'xxx')

print(ret2.__dict__)

# object
object

# 可变变量和不可变变量的操作
## 可变变量 list: 当 name = 【】实际是在内存中建立了一个空间，list类型在实例中不能修改，但该
## list空间里的内容是可以变的，这是list的属性，也可以把这种看作一个list的实例，我们可以调用
## list类的方法
class A:
    name = []

p1 = A()
p2 = A()
p1.name.append(1)

p1.age = 12
p1.name，p2.name，A.name 分别又是什么？为什么？
print(p1.age)
print(p2.age)
print(A.age)
p1.name，p2.name，A.name 分别是什么？
print(p1.name)
print(p2.name)
print(A.name)

## 不可变变量的，不能变 
class A:
    name = 'alex'

p1 = A()
p2 = A()
p1.name = 'wusir'
print(p1.name)
print(A.name）

# 狭义封装
class Parent:
    def __func(self):
        print('in Parent func')

    def __init__(self):
        self.__func()

class Son(Parent):
    def __func(self):
        print('in Son func')

son1 = Son()

class Test: 
    def func(self): 
        print("Is testing")

    def __init__(self, a): 
        self.func() # 在类中—__init__会执行代码
        self.a = a

test = Test(10)

# *的意义
def star(*args, **kwargs):
    print(args)
    print(*args) # 函数的执行，print的执行
    print(kwargs)
    print(*kwargs)
    print(**kwargs) # print中没有关键字错误，所以报错

star(1, 2, 3, var1=4, var2=5 , var3=6)

print(*{"var1":1, "var2":2, "var3":3}))
print(*[1, 2, 3])

star([1, 2, 3], [4, 5, 6])
star(*[1, 2, 3], *[4, 5, 6])

print(a = 1) # **kwargs的print，print中并没有a这个关键字

# _的作用，分割数字
print(10_000_000)

# 面向对象总结
## type
class A:
    # def __new__(cls)
    pass
a = A()
type(a)
type(A)
type(object)

## 类的加载顺序
class Person:
    ROLE = "China"
    print(ROLE)

## 在任何类中调用的方法,都要自习分辨一下这个self到低是谁的对象
class Foo:
    print("In Foo")
    def __init__(self):
        self.func()

    def func(self):
        print("IN Foo.func")

class Son(Foo):
    def func(self):
        print("IN Son.func")

s = Son()

# 广度优先,super及mro演示
class A:
    def func(self):
        print("In A")

class B(A):
    def func(self):
        super().func()
        print("In B")

class C(A):
    def func(self):
        super().func()
        print("In C")

class D(B, C):
    def func(self):
        super().func()
        print("In D")

d = D()
d.func()
b = B()
b.func()
D.mro()

## 反射
class Student:
    ROLE = "STUDENT"
    def __init__(self, name):
        self.name = name
    @classmethod
    def check_course(self):
        print("check_course")
    def chose_course(self):
        print("chose_course")
    def choosed_coures(self):
        print("查看已选择的课程")
    @staticmethod
    def login():
        print("登录")

st = Student("alex")
name = getattr(st, "name")
getattr(st, "login")()
getattr(Student, "login")()
getattr(Student, "ROLE")
getattr(Student, "check_course")

command = input(">>>>>")
if hasattr(st, command):
    getattr(st, command)()

## 装饰器
import time 

# 为函数增加功能：改变了源代码, 也改变了调用方式，不是func()
def func(): 
    time.sleep(2)
    print("睡着了")

def func2():
    start_time = time.time()
    time.sleep(2)
    print('睡着了')
    end_time = time.time()
    run_time = end_time - start_time
    print("持续了{}秒".format(run_time))


func2()

# 方法二 使用其他函数 为函数增加功能，不修改源代码，但是调用方式改变了
import time

def func(): 
    time.sleep(2)
    print("睡着了")

def timer(func):
    start_time = time.time()
    func()
    end_time = time.time()
    run_time = end_time - start_time
    print("持续了{}秒".format(run_time))

timer(func) # 本身的调用方式为func()

# func = timer
# func() 报错达到了递归的极限，为什么

# 方法三 没修改源码 加了功能 且 调用方式不变，但结果不对
import time

def func(): 
    time.sleep(2)
    print("睡着了")

def timer(func):
    start_time = time.time()
    func()
    end_time = time.time()
    run_time = end_time - start_time
    print("持续了{}秒".format(run_time))
    return func

func = timer(func)
func()

# 方法四 实现了目的
import time

def func(): 
    time.sleep(2)
    print("睡着了")

def timer(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        run_time = end_time - start_time
        print("持续了{}秒".format(run_time))
    return wrapper

func = timer(func)
func()

# 新要求：为非func函数增加功能
import time

def func(): 
    time.sleep(2)
    print("睡着了")

def timer(func):
    def wrapper(*args, **kwargs): # 通过动态参数，将不同函数的函数传入
        start_time = time.time()
        res = func(*args, **kwargs) # 原函数的功能
        end_time = time.time()
        run_time = end_time - start_time
        print("持续了{}秒".format(run_time))
        return res # 返回func的结果
    return wrapper

func = timer(func)
func()

@timer  # 等价于 func2 = timer(func2) func2()
def func2():
    time.sleep(3)
    print("睡着了")

func2()

# 语法糖
class Myclass:
    def __init__(self,string):
        self.string = string
    
    def __add__(self, other): # 计算两个字符串中有多少个*
        res = self.string.count("*") + other.string.count("*")
        return res

obj1 = Myclass("123***456***")
obj2 = Myclass("12**34**")
star_count = obj1 + obj2
star_count2 = obj1.__add__(obj2)
print("语法糖：{}\n普通方法：{}".format(star_count, star_count2))

# 双下call方法
class T_call:
    def __call__(self, *args, **kwargs):
        print('执行call方法')

a = T_call()
a() # 相当于调用__call__方法
a.__call__()

# 内置方法__len__
class mylist:
    def __init__(self):
        self.lst = [1,2,3,4,5,6]
        self.name = 'alex'
        self.age = 83
    def __len__(self):
        print('执行__len__了')
        return len(self.__dict__) # self.__dict__返回的使一个dict，这里相当于调用了dict的len

l = mylist()
print(len(l))

class String:
    def __init__(self, string):
        self.string = string
    
    def __len__(self):
        return len(self.string)

string = String("1234567")
len(string)

# __new__
class Test:
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls) 
        print("在new方法中", obj)
        return obj # 将开辟的空间返回，意味着把obj传入到了self中，如果没有return，那么就不会进行init
    
    def __init__(self):
        print("在inint中", self)

test = Test()

# 单例类
class Single:
    ROM = None
    def __new__(cls, *args, **kwargs):
        if not cls.ROM:
            cls.ROM = object.__new__(cls)
            # cls.ROM = True 这里不饿能这么实现，因为__new__的返回值必须是个空间地址
        return cls.ROM
    
    def __init__(self, name):
        self.name = name
        print("内存空间{}".format(self.name), self)

single = Single("single")
single2 = Single("single2")


class Single:
    ROM = None
    def __new__(cls, *args, **kwargs):
        if not cls.ROM:
            obj = object.__new__(cls)
            # cls.ROM = object.__new__(cls)
            cls.ROM = not None 
        return obj # 这里不能这么实现，因为第一次创建对象之后，ROM变为非空，
                   # 所以第二次创建的时候就不会执行object.__new__, 所以obj不会被赋值
    
    def __init__(self, name):
        self.name = name
        print("内存空间{}".format(self.name), self)

single = Single("single")
single2 = Single("single2")

# 一个类
# 对象的属性 : 姓名 性别 年龄 部门
# 员工管理系统
# 内部转岗 python开发 - go开发
# 姓名 性别 年龄 新的部门
# alex None 83 python
# alex None 85 luffy

# 1000个员工
# 如果几个员工对象的姓名和性别相同,这是一个人
# 请对这1000个员工做去重

class Employee:
    # 因为set中也包含hash和eq方法，所以通过定义类的hash和eq就可以完成去重
    # 当set中的hash和eq被调用时，因为Employee类中也有这些方法，按照属性调用顺序按照命名空间的原则，
    # 会先调用Employee的hash和eq
    def __init__(self, name, sex, age, partment):
        self.name = name
        self.sex = sex
        self.age = age
        self.partment = partment
    def __hash__(self): # 自定义hash，使hash对员工的姓名和性别进行hash
        return hash("{name}{sex}".format(name = self.name, sex = self.sex))
    def __eq__(self, other): # 自定义eq, 确保员工的姓名和性别除了hash一致外，值也相等
        if self.name == other.name and self.sex == other.sex:
            return True 

employee_lst = []

for i in range(200):
    employee_lst.append(Employee("alex", "male", i, "python"))
for i in range(200):
    employee_lst.append(Employee("wusir", "male", i, "python"))
for i in range(200):
    employee_lst.append(Employee("taibai", "female", i, "python"))

lst_clear = set(employee_lst)
