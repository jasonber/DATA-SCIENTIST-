import os
import sys

file_path = __file__.split('/')
base_path = '/'.join(file_path[:-2])

sys.path.append(base_path)
# print("file_path:{}\nbase_path:{}\n".format(file_path, base_path))
from core import main


if __name__ == '__main__':
    main.working()