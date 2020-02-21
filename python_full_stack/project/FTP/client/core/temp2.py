import os

print("文件地址", __file__)
file_path = "\\".join(__file__.split(r'/')[:-1])
print("修改文件地址", file_path)
os.chdir(file_path)
print("当前地址", os.getcwd())

import temp