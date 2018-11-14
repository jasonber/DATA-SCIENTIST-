import numpy as np
from math import log
import operator
"""
决策树的算法的实现分为两部分
1、利用已有的信息构建决策树
    a、树的构建需要id3， c4.5， cart算法。主要目的是向着数据混乱程度变小的方向画树
    b、如何保存建立好的树
2、使用建立好的决策数去进行分类
    a、如何给出分类标签
"""
def entropy(dataset):
    # 获得样本量
    num_entries = len(dataset)
    label_num = {}
    for feature_vector in dataset:
        current_label = feature_vector[-1]
        if current_label not in label_num.keys():
            label_num[current_label] = 0
        label_num[current_label] += 1

    ent = 0.0
    for key in label_num:
        probability = float(label_num[key]) / num_entries
        # 因为循环的存在 所以自增就能完成求和
        # 因为熵是概率和log乘积的负值，所以使用自减
        ent -= probability * log(probability, 2)

    return ent


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, feature, value):
    """
    按照给定的特征划分数据集
    :param dataset:
    :param feature: 分割的特征
    :param value:
    :return:
    """
    return_dataset = []
    for feature_vector in dataset:
        if feature_vector[feature] == value:
            # 以value为分界点 将某个特征分成两部分
            reduce_feature_vector = feature_vector[:feature]
            reduce_feature_vector.extend(feature_vector[feature + 1:])
            return_dataset.append(reduce_feature_vector)
    return return_dataset


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    # 遍历每个特征
    for i in range(num_features):
        feature_value_list = [example[i] for example in dataset]
        # 获取每个特征中包含的值，并使这些值唯一
        unique_feature_value = set(feature_value_list)
        new_entropy = 0.0
        for value in unique_feature_value: # 这部分不太懂了
            sub_dataset = split_dataset(dataset, i, value)
            probability = len(sub_dataset) / float(len(dataset)) # 这里计算的是该类别在全部数据集中概率
            new_entropy += probability * entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def voting(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 排序完成后将字典转换为了 tuple组成的list
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return voting(class_list)
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}

    # 保证删除best_feature不会对 原始的数据造成影响
    sub_labels = labels[:]
    del(sub_labels[best_feature])

    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        # 递归 要死记硬背了
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value),sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vector):
    first_node = list(input_tree.keys())[0]
    second_dict = input_tree[first_node]
    feat_index = feat_labels.index(first_node)
    for key in second_dict.keys():
        if test_vector[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vector)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, tree_file):
    import pickle
    # 不适用with as
    # fw = open(tree_file, 'w')
    # pickle.dump(input_tree, fw)
    # fw.close()
    # https://blog.csdn.net/qq_33363973/article/details/77881168
    # type error
    # with open(tree_file, 'w') as fw:
    with open(tree_file, 'wb') as fw:
        pickle.dump(input_tree, fw)


def load_tree(tree_file):
    import pickle
    fr = open(tree_file, 'rb')
    return pickle.load(fr)
