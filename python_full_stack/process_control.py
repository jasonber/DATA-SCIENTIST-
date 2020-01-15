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