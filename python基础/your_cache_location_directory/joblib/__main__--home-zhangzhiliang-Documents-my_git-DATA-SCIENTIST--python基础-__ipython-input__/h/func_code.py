# first line: 1
@memory.cache
def h(x):
    print('A second long-running calculation, using g(x)')
    return np.vander(x)
