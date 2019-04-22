'''
# 为什么使用joblib 
## 管道的好处 
管道系统能够提供以下有用的特性：  
**数据流编程的性能:**  
按需进行计算：在管道系统中，输出是由输入的需要决定的。  
透明的并行过程：函数编成，可自由调整结果。  
**从头到尾追踪代码**  
追踪数据和计算：能重复计算过程   
监视数据流：检查中间结果，帮助调试和理解代码   
***管道的缺陷***  
Joblib的理念是保证底层算法代码不变，避免框架的调整   
## Joblib的方法  
函数是每个人都会使用的简单方法。Joblib中的任务可以写成函数。  
以有意义的方式跟踪参数需要指定数据模型。Joblib放弃了这种方法，并使用散列来提高性能和健壮性。   
## 设计选择
只依赖python  
健壮、测试好的代码、以功能为代价  
在对大数据集进行科学计算时不需要改变原数据，就可以很快很好的完成科学计算。
'''
'''
# 按需进行验算：memory 类   
案例
memory类使函数的参数评估变得省事，通过存储结果，默认使用硬盘，以及不重复运行相同参数，来实现。  
joblib将输出明确的保存在一个文件中，并且它被设计与非哈希和潜在大量输入输出的数据类型。如numpy的数组。
'''
# 简单的示例
# 定义一个缓存路径
cachedir = 'your_cache_location_directory'
# 为这个缓存路径声明一个内存上下文中,进行初始化
from joblib import Memory
memory = Memory(cachedir, verbose=0)
# 经过以上的初始化，将memory装饰为函数将它的输出结果存入到上下文中
@memory.cache
def f(x):
    print('Running f(%s)'%x)
    return x
# 用相同的参数调用两次这个函数，第二次不执行，从缓存字典中加载输出结果
print(f(1))
print(f(1))
# 然而使用不同的参数调用函数，将重新计算结果
print(f(2))

'''与memoize的比较
memoize装饰器缓存一个函数的所有输入和输出结果。因此它能通过少的开销，避免同一函数运行两次
不过，它每次调用时都会将输入对象与缓存中的对象进行对比。因此，当输入对象较大时会产生巨大的开销。
并且，这一方法并不适用于numpy数组，或其他受非显著波动的对象。最后，使用大的对象的memoize
将消耗所有的内存，如果使用Memory，对象始终存储在硬盘上，对速度和内存的使用有一个更为持久的
优化（joblib.dump()）。

总而言之，memoize最适合小的输入和输出对象，Memory最适合复杂的输入和输出对象以及磁盘的持久保存

'''

'''在numpy中的使用
Memory最初的动机是在numpy数组上使用类似模式的存储。Memory使用更快的加密hash来检查输入参数
是否已经被计算
'''
# 举例
# 定义两个函数：第一个函数使用一个数字作为参数，输出一个数组，并被第二个使用，两个函数均使用
# Memory.cache来装饰
import numpy as np
@memory.cache
def g(x):
    print('A long-running calcultation, with parameter %s'%x)
    return np.hamming(x)

@memory.cache
def h(x):
    print('A second long-running calculation, using g(x)')
    return np.vander(x)
