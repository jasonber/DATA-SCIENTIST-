# first line: 1
@memory.cache
def g(x):
    print('A long-running calcultation, with parameter %s'%x)
    return np.hamming(x)
