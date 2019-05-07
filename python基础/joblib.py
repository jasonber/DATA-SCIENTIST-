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

# 如果使用数组调用函数h并传给函数g，那么h不再重复运行
a = g(3)
b = h(a)
b2 = h(a)
np.allclose(b, b2)
'''
使用memmapping
Memapping内存中使用map，当加载较大的numpy数组时，加速缓存查询
'''
cachedir2 = 'your_cachedir2_location'
memory2 = Memory(cachedir2, mmap_mode='r')
square = memory2.cache(np.square)
a = np.vander(np.arange(3)).astype(np.float)
square(a)
'''
提示：在上述例子中进行debug需要注意。对于追踪那些被重复执行流程，以及消耗的时间
'''
'''
如果square使用相同的参数，那么它返回的值将通过memmapping从磁盘中加载
'''
# memory 感觉更像是存储在硬盘上的函数
res = square(a)
print(repr(res))
'''
在windows中，memmap必须关闭以避免文件被锁；使用 del 关闭numpy.memmap这会在磁盘作出变化
'''
del res
'''
提示：
如果memory mapping模式使用r，就像上面的例子，数组将只能读取，而不能修改
另一方面，使用‘r+’或者‘w+’将能改变数组，但这一改变会发生在硬盘上，将破坏缓存。如果
你想在内存中改变数组，我们建议你使用‘c’：写入时进行备份。
'''
'''搁置：为缓存中的值设置引用
在一些例子中，为缓存设置引用非常有用，这可以替代调用结果本身。一个经典的例子，把一个大的数组
分发给不同的工作者：引用可以代替数据本身在网络上传递，向joblib的缓存中发送一个引用，让工作者
从网络的文件系统中读取数据，也能潜在的节省一些系统级别的内存。
’‘’
‘’‘
使用方法call_and_shelve为包装好的函数设置引用
'''
result = g.call_and_shelve(4)
result
'''
一旦计算，g的输出结果将保存在硬盘上， 并且从内存中删除。
获取相关的值可以使用方法 get。
'''
result.get()
'''
可以使用方法clear清除这个值的缓存。调用clear会清楚硬盘上的值。任何get的调用将引发
关键错误
'''
result.clear()
result.get()
'''
一个MemorizedResult 实例 包含所有它读取缓存所需的所有内容。该实例可以被处理以进行传输或存储，
并且可以被打印，甚至复制到其他的python解释器中
'''
'''
当缓存不可用时，使用搁置
当缓存不可用时（如Memory（None）），call_and_shelve将返回NotMemorizeResult实例，
将存储所有的函数输出，而不仅仅是引用（因为没有指向任何内存）。除了复制粘贴功能之外，上诉
所有功能都是有效的
'''
'''
陷阱
在session之间，通过函数名来识别函数
因此如果对不同的函数赋予相同的名字，那么他们将会互相覆盖缓存（如名称冲突），并且会发生
不希望出现的重复运行
'''
@memory.cache
def func(x):
    print('Running func(%s)'%x)

func2 = func

@memory.cache
def func(x):
    print('Running a different func(%s)'%x)

# jobib8.0及以上在调用相同的session时,尽管joblib会警告你，不会发生名称冲突
func(1)
func2(1)

# 一旦相同的session被调用，就不再进行重复计算了。
func(1)
func2(1)

'''lambda 函数
注意在python2.7中lambda函数是不能被分开的'''
def my_print(x):
    print(x)
f = memory.cache(lambda:my_print(1))
g = memory.cache(lambda:my_print(2))
f()
f()
g()
g()
f()

'''在某些复杂的对象上，不可能使用memory，如可调用对象a__call__'''
# 然而，它可以在numpy上运行
sin = memory.cache(np.sin)
print(sin(0))

'''缓存方法：memory是为纯函数设计的，并且不推荐它在methond上使用
如果想在class中使用cache，建议对纯函数使用cache，并在class中使用，如下所示：
'''
@memory.cache
def compute_func(arg1, arg2, arg3):
    return result

class Foo(object):
    def __inti__(self, args):
        self.data = None

    def compute(self):
        self.data = compute_func(self.arg1,self.arg2, 4)

'''
不推荐对方法使用Memory，从维护的角度来讲并且存在一些缺陷。因为在软件升级的时候，很容易
忘记这些警告。如果不能避免，那么请记住下面这些缺陷：
'''
# 1、一个方法不能在class定义时被decorated，因为实例化一个class时，第一个参数被限制为
# self，它不能轻易的成为Memory的对象。下面的代码不能运行：
class Foo(object):

    @memory.cache # wrong
    def method(self, args):
        pass
# 在实例化时decorate的正确方法：
class Foo(object):

    def __init__(self, args):
        self.method = memory.cache(self.method)

    def method(self,...):
        pass
# 2、cache方法有一个self参数。这意味着如果变换self，那么结果将会重复计算。例如
# 如果self.attr变为调用self.method, 那么即使self.method不使用self.attr，也会
# 重新计算结果。结果是，在后面的调用中，不会再使用self.method创建的cache。如果知道
# self.method的结果并不依赖self,那么可以通过
# self.method = memory.cache(self.method, ignore=['self'])来解决这个问题。

'''
忽略某些参数
当某些参数改变时（如调式标志），忽略参数可能对不进行重复计算是有用的。
'''
@memory.cache(ignore=['debug'])
def my_func(x, debug=True):
    print('Called with x= %s'%x)
my_func(0)
my_func(0, debug=False)
my_func(0, debug=True)
