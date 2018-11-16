# 详细讲解 https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/9.RegTrees/treeExplore.py

from tkinter import *
import numpy as np
import reg_tree
import matplotlib
# TkAgg使得tkinter可以调用matplotlib
# Matplotlib文件并设定后端为TkAgg
matplotlib.use('TkAgg')
# 两个import声明将TkAgg和Matplotlib图链接起来。
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


path = "/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/" \
       "machine_learing_algorithm/machine_learning_in_action/9_cart_tree/sine.txt"

def re_draw(stop_s, stop_n):
    # 清空画布
    # 清空之前的图像，使得前后两个图像不会重叠。
    re_draw.f.clf()
    # 清空时图像的各个子图也都会被清除,所以要重建画布
    re_draw.a = re_draw.f.add_subplot(111)
    # .get()选中了控件
    if check_button_var.get():
        if stop_n < 2:
            stop_n = 2
        my_tree = reg_tree.create_tree(re_draw.raw_data, reg_tree.model_leaf,
                                       reg_tree.model_error, (stop_s, stop_n))
        y_predict = reg_tree.create_forecast(my_tree, re_draw.test_data,
                                             reg_tree.model_tree_value)
    else:
        my_tree = reg_tree.create_tree(re_draw.raw_data, ops=(stop_s, stop_n))
        y_predict = reg_tree.create_forecast(my_tree, re_draw.test_data)
    # 真实值采用scatter()方法绘制，代表真实情况。而预测值则采用plot()方法绘制，代表预测情况
    # 这是因为scatter()方法构建的是离散型散点图，而plot()方法则构建连续曲线。
    re_draw.a.scatter(re_draw.raw_data[:, 0].A, re_draw.raw_data[:, 1].A, s=5)
    re_draw.a.plot(re_draw.test_data, y_predict, linewidth=2.0, color='red')
    re_draw.canvas.draw()


def get_input():
    # 如果Python可以把输入文本解析成整数就继续执行，
    # 如果不能识别则输出错误消息，同时清空输入框并恢复其默认值
    try:
        stop_n = int(stop_n_entry.get())
    except:
        stop_n = 10
        print("enter Integer for stop_n")
        stop_n_entry.delete(0, END)
        stop_n_entry.insert(0, '10')
    try:
        stop_s = float(stop_s_entry.get())
    except:
        stop_s = 1.0
        print("enter Float for stop_s")
        stop_s_entry.delete(0, END)
        stop_s_entry.insert(0, '1.0')
    return stop_n, stop_s


def draw_new_tree():
    stop_n, stop_s = get_input()
    re_draw(stop_s, stop_n)


root = Tk()

Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)

Label(root, text='stop_n').grid(row=1, column=0)
# Entry部件是一个允许单行文本输入的文本框
stop_n_entry = Entry(root)
stop_n_entry.grid(row=1, column=1)
stop_n_entry.insert(0, '10')
Label(root, text="stop_s").grid(row=2, column=0)
stop_s_entry = Entry(root)
stop_s_entry.grid(row=2, column=1)
stop_s_entry.insert(0, '1.0')
# 通过设定columnspan和rowspan的值来告诉布局管理器是否允许一个小部件跨行或跨列
Button(root, text="ReDraw", command=draw_new_tree).grid(row=1, column=2, rowspan=3)
# 为了读取Checkbutton的状态需要创建一个变量，也就是IntVar。
check_button_var = IntVar()
check_button = Checkbutton(root, text='Model Tree', variable=check_button_var)
check_button.grid(row=3, column=0, columnspan=2)

re_draw.raw_data = np.mat(reg_tree.load_dataset(path))
re_draw.test_data = np.arange(min(re_draw.raw_data[:, 0]),
                              max(re_draw.raw_data[:, 0]), 0.01)
# 创建一个画板 canvas
re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

re_draw(1.0, 10)
root.mainloop()
