from time import time

# Lazy evaluation
t = time()

abbreviations = ['cf.', 'e.g', 'ex.', 'etc.', 'fig.', 'i.e.', 'Mr.', 'vs.']
for i in range(1000000):
    for w in ('Mr.', 'Hat','is', 'cashing', 'the', 'black', 'cat', '.'):
        # if w in abbreviations:
        if w[-1] == '.' and w in abbreviations:
            pass

print('total run time {}'.format(time() - t))

a = [1, 2, 3, 4, 5]

def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a+b



