import os
file_path = '/'.join(__file__.split('/')[:-1])
print(__file__)
print(file_path)
os.chdir(file_path)
import sys
sys.path.append(file_path)
from ..core import main2

if __name__ == '__main__':
    mian2.main()