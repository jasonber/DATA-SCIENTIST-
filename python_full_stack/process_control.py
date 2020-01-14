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

