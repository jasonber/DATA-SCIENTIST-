import numpy as np
from math import log
import operator

def entropy(dataset):
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
            reduce_feature_vector = feature_vector[:feature]
            reduce_feature_vector.extend(feature_vector[feature + 1:])
            return_dataset.append(reduce_feature_vector)
    return return_dataset


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        unique_feature = set(feature_list)
        new_entropy = 0.0
        for value in unique_feature: # 这部分不太懂了
            sub_dataset = split_dataset(dataset, i, value)
            probability = len(sub_dataset) / float(len(dataset)) # 这里计算的是该类别出在全部数据集中概率
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

    # del(labels[best_feature]) 这里是错误的 需要自己再消化下
    sub_labels = labels[:]
    del(sub_labels[best_feature])

    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value),sub_labels)
    return my_tree
