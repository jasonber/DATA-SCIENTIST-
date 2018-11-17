"""
logistics 回归的重点
1、sigmoid函数
2、目标函数
3、优化算法：梯度下降算法（推荐）、正规方程
"""

import numpy as np
from itertools import product
path = "/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/" \
       "machine_learning_in_action/5_Logistic/testSet.txt"

def load_dataset():
    data_matrix = []
    label_matrix = []
    fr = open(path)
    for line in fr.readlines():
        line_array = line.strip().split()
        # python真的是太灵活了
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix


def sigmoid(input_x):
    return 1.0 / (1+ np.exp(-input_x))



def bath_GA(input_data, class_label):
    data_matrix = np.mat(input_data)
    # 标签由行向量变为列向量
    label_matrix = np.matrix(class_label).transpose()
    r, c = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    # 权重列向量
    weights = np.ones((c, 1))
    # 擦logistics的目标函数忘了。。。。。
    for k in range(max_cycles):
        y_hat = sigmoid(data_matrix * weights)
        error = (label_matrix - y_hat)
        # 矩阵的乘法是有运算顺序的，从右到左操作，
        # 参数的调整
        weights = weights + alpha  * data_matrix.transpose() * error
    return weights


# 梯度算法详解：
# https://blog.csdn.net/legend_hua/article/details/80633525
def stochastic_GA(data_matrix, class_labels):
    r, c = np.shape(data_matrix)
    alpha = 0.01
    # 权重标量
    weights = np.ones(c)
    for i in range(r):
        y_hat = sigmoid(np.sum(data_matrix[i] * weights))
        # 直接调用标签的值
        error = class_labels[i] - y_hat
        weights = weights + alpha * error * data_matrix[i]
    return weights


def sto_GA_update(data_matrix, class_labels, num_iter=150):
    r, c = np.shape(data_matrix)
    weights = np.ones(c)
    for j in range(num_iter):
        data_index = list(range(r))
        for i in range(r):
            # 模拟退火算法，越迭代，学习率越小
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机梯度的精髓，随机选取值。做到在梯度方向的跨越式变化
            random_index = int(np.random.uniform(0, len(data_index)))
            y_hat = sigmoid(sum(data_matrix[random_index] * weights))
            error = class_labels[random_index] - y_hat
            weights = weights + alpha * error *data_matrix[random_index]
            del(data_index[random_index])
    return weights

def decision_boundary(weight):
    import matplotlib.pyplot as plt
    # 这是个是啥
    if type(weight).__name__=="matrix":
        weights = weight.getA()
    else:
        weights = weight
    data_matrix, label_matrix = load_dataset()
    data_array = np.array(data_matrix)
    r = np.shape(data_array)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(r):
        if int(label_matrix[i]) == 1:
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])
    fig = plt.figure(figsize=(6, 8))
    ax =  fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    """
    决策边界是一条直线，也是超平面的基础。就是将数据分开的东西。
    在logistic中，矩阵x × theta转置=0 为决策边界。theta为weights组成的向量
    y的由来
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def classify_vector(input_x, weights):
    prob = sigmoid(sum(input_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    path_train ="/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/" \
                "machine_learing_algorithm/machine_learning_in_action/5_Logistic/horseColicTraining.txt"
    path_test = "/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/" \
                "machine_learing_algorithm/machine_learning_in_action/5_Logistic/horseColicTest.txt"

    with open(path_train) as fr_train:
    #fr_train = open("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
     #               "machine_learing_algorithm/machine_learning_in_action/5_Logistic/horseColicTraining.txt")
        train_set = []
        train_label = []
        for line in fr_train.readlines():
            cursor_line = line.strip().split('\t')
            line_array = []
            for i in range(21):
                line_array.append(float(cursor_line[i]))
            train_set.append(line_array)
            train_label.append(float(cursor_line[21]))
    train_weights = sto_GA_update(np.array(train_set), train_label, 500)
    # fr_train.close()

    error_count = 0
    num_test_vector = 0.0
    with open(path_test) as fr_test:
    # fr_test = open("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
    #                "machine_learing_algorithm/machine_learning_in_action/5_Logistic/horseColicTest.txt")
        for line in fr_test.readlines():
            num_test_vector += 1.0
            cursor_line = line.strip().split('\t')
            line_array = []
            for i in range(21):
                line_array.append(float(cursor_line[i]))
            if int(classify_vector(np.array(line_array), train_weights)) != int(cursor_line[21]):
                error_count += 1
    error_rate = (float(error_count) / num_test_vector)
    print("the error rate of this test is:{:f}".format(error_rate))
    # fr_test.close()
    return error_rate


def multiple_test():
    num_test = 10
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colic_test()
    print("after {:d} iterations the average error rate is: {:f}".format(num_test, error_sum / float(num_test)))

