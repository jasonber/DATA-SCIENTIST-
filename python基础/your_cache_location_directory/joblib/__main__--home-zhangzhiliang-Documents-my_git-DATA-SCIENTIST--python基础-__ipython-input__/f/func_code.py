# first line: 1
@memory.cache
def f(x):
    print('Running f(%s)'%x)
    return x
