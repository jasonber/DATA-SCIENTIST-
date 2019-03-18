import numpy as np
import operator

def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(in_x, data_set, labels, k):
    '''

    :param in_x: 分类的数据，向量
    :param data_set: 训练集
    :param labels: 训练集的标签
    :param k: 分为几类
    :return: 降序排列，返回第一个值。即发生率最高的值
    '''
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 计算距离
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    # 获取距离列表的升序结果的索引
    sorted_dist_indicies = distance.argsort()
    # 进行投票，并将投票次数放入到分类dic中
    class_count = {}
    for i in range(k):
        vote_ind_label = labels[sorted_dist_indicies[i]]
        class_count[vote_ind_label] = class_count.get(vote_ind_label, 0) + 1
    # 对投票结果从大到小排序，sorted默认为升序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回排序后的第一个值，也就是投票最多的类别
    return sorted_class_count[0][0]
