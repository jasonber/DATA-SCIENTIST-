import numpy as np

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

def grad_ascent(input_data, class_label):
    data_matrix = np.mat(input_data)
    label_matrix = np.matrix(class_label).transpose()
    r, c = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((c, 1))
    # 擦logistics的目标函数忘了。。。。。
    for k in range(max_cycles):
        y_hat = sigmoid(data_matrix * weights)
        error = (label_matrix - y_hat)
        # 矩阵的乘法是有运算顺序的，从右到左操作，
        # 参数的调整
        weights = weights + alpha  * data_matrix.transpose() * error
    return weights


def decision_boundary(weight):
    import matplotlib.pyplot as plt
    # 这是个是啥
    weights = weight.getA()
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
