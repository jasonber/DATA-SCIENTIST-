import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

# 测试
group, labels = kNN.createDataSet()

kNN.classify0([0, 0], group, labels, 3)

# 约会分类
dating_data , dating_labels = kNN.file_to_matrix("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                                                 "machine_learing_algorithm/machine_learning_in_action/2_KNN/datingTestSet2.txt")

fig = plt.figure(figsize=(8, 6))
# 如何添加图例
plt.scatter(dating_data[:, 1], dating_data[:, 2], 15.0*np.array(dating_labels), 15.0*np.array(dating_labels))
plt.xlabel("time of gaming")
plt.ylabel("litre of ice-cream")
plt.show()

# 实现图例
# https://blog.csdn.net/xiaobaicai4552/article/details/79069207
# 3个分类，就有3组xy

fig2 = plt.figure()
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []

for i in range(len(dating_labels)):
    if dating_labels[i] == 1:
        type1_x.append(dating_data[i][0])
        type1_y.append(dating_data[i][1])
    if dating_labels[i] == 2:
        type2_x.append(dating_data[i][0])
        type2_y.append(dating_data[i][1])
    if dating_labels[i] == 3:
        type3_x.append(dating_data[i][0])
        type3_y.append(dating_data[i][1])

# mpl.rcParams['font.sans-serif'] = ['FangSong']
# mpl.rcParams['axes.unicode_minus'] = False
plt.scatter(type1_x, type1_y, s=20, c='r', label='don`t like')
plt.scatter(type2_x, type2_y, s=40, c='b', label='just so so')
plt.scatter(type3_x, type3_y, s=60, c='k', label='great')

plt.legend()
plt.show()

# 面向对象的可视化方法
fig3 = plt.figure(figsize=(8, 6))
axes = plt.subplot(111)

type1 = axes.scatter(type1_x, type1_y, s=20, c='r')
type2 = axes.scatter(type2_x, type2_y, s=40, c='b')
type3 = axes.scatter(type3_x, type3_y, s=60, c='k')

plt.legend((type1, type2, type3), ('don`t like', 'just so so', 'great'))
plt.show()

# 使用get_legend_handles_labels()
fig4 = plt.figure(figsize=(8, 6))
axes2 = plt.subplot(111)
type11 = axes2.scatter(type1_x, type1_y, s=20, c='r', label='don`t like')
type21 = axes2.scatter(type2_x, type2_y, s=40, c='b', label='just so so')
type31 = axes2.scatter(type3_x, type3_y, s=60, c='k', label='great')
handles, lables = axes2.get_legend_handles_labels()
# axes2.legend(handles, lables)
plt.legend()
plt.show()

norm_mat, ranges, min_vals = kNN.auto_norm(dating_data)


