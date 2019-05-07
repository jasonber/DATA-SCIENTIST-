# first line: 1
@memory.cache(ignore=['debug'])
def my_func(x, debug=True):
    print('Called with x= %s'%x)
