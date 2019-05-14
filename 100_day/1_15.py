# 水仙花数字
import numpy as np

def Nar_num():
    for i in range(100, 1000):
        a = int(i / 100)
        b = int((i - a * 100) / 10)
        c = int(i - a * 100 - b * 10)
        num = np.power(a, 3) + np.power(b, 3) + np.power(c, 3)
        if num == i:
            print('%d is Nar num' % (i))
        # else:
            # print('%d not Nar name' % (i))

if __name__ == '__main__':
    Nar_num()

# 完美数字
import time 
import math 

start = time.clock()
for num in range(1, 1000):
    sum = 0
    for factor in range(1, int(np.sqrt(num)) + 1):
        if num % factor == 0:
            sum += factor
            if factor > 1 and num / factor != factor:
                 sum += num / factor
    if sum == num:
        print(num)

end = time.clock()
print('执行时间', (end - start), '秒')