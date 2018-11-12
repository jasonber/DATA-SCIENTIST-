import numpy as np
import operator
import os
import datetime

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    :param inX: test set
    :param dataSet: train set
    :param labels:
    :param k: number of nearest element
    :return: label
    """

    dataSetSize = dataSet.shape[0]
    # 将样本数据转换为训练集shape的矩阵
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 获取值排序后的索引顺序
    sortedDistIndicies = np.argsort(distances)
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 记录距离最小的前K个类，并存放入列表。KEY对应标签，VALUE对应计数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file_to_matrix(filename):
    file = open(filename)
    array_of_lines = file.readlines()
    number_of_lines = len(array_of_lines)
    matrix = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        # 去掉换行符
        line = line.strip()
        # 用tab分割，整理数据格式
        list_from_line = line.split('\t')
        matrix[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return matrix, class_label_vector


def auto_norm(dataset):
    """
    极差归一
    :param dataset:
    :return:
    """
    min_vals = dataset.min(0)  # 这是个向量 numpy.min(0)表示列中最小 1表示行中最小
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(dataset.shape)
    m = dataset.shape[0]
    # 进行矩阵运算
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals

def dating_class_test():
    test_size = 0.1
    dating_data, dating_labels = file_to_matrix("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                                                 "machine_learing_algorithm/machine_learning_in_action/2_KNN/datingTestSet2.txt")
    norm_data, ranges, min_vals = auto_norm(dating_data)
    row = norm_data.shape[0]
    test_set_len = int(row * test_size)
    error_count = 0.0

    for i in range(test_set_len):
        classifier_result = classify0(norm_data[i,:], norm_data[test_set_len:row], dating_labels[test_set_len:row], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]):
            error_count += 1
    print("the total error rate is: {:f}".format(error_count / float(test_set_len)))

def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    time_of_gaming = float(input('percentage of time spent playing video game?'))
    miles_of_airfly = float(input("frequent flier miles earned per year?"))
    liters_of_icecream = float(input("liters of ice cream consumed per year?"))
    dating_data, dating_labels = file_to_matrix("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                                                "machine_learing_algorithm/machine_learning_in_action/2_KNN/datingTestSet2.txt")
    norm_data, ranges, min_vals = auto_norm(dating_data)
    information = np.array([time_of_gaming, miles_of_airfly, liters_of_icecream])
    result = classify0((information - min_vals) / ranges, norm_data, dating_labels, 3)
    print("You will probably like this person:{}".format(result_list[result - 1 ]))

def img_to_vector(filename):
    return_vector = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
       line = file.readline()
       for j in range(32):
            return_vector[0, 32 * i + j] = int(line[j])
    return return_vector

def hand_write_class_test():
    starttime = datetime.datetime.now()

    hand_write_labels = []
    train_path = "/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/" \
                 "machine_learning_in_action/2_KNN/digits/trainingDigits"
    test_path = "/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/" \
                "machine_learning_in_action/2_KNN/digits/testDigits"
    train_file_list = os.listdir(train_path)
    train_num_of_digits = len(train_file_list)
    train_data = np.zeros((train_num_of_digits, 1024))
    for i in range(train_num_of_digits):
        file_name_str = train_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_digits_str = int(file_str.split('_')[0])
        hand_write_labels.append(class_digits_str)
        train_data[i, :] = img_to_vector('{}/{}'.format(train_path, file_name_str))

    test_file_list = os.listdir(test_path)
    error_count = 0.0
    test_num_of_digits = len(test_file_list)
    for i in range(test_num_of_digits):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_digits_str = int(file_str.split('_')[0])
        test_data = img_to_vector('{}/{}'.format(test_path, file_name_str))
        result = classify0(test_data, train_data, hand_write_labels, 3)
        print("the classifier came back with: {:d}, the real answer is: {:d}".format(result, class_digits_str))
        if (result != class_digits_str):
            error_count += 1.0
    print("the total number of errors is {}".format(error_count))
    print("the total number of error rate is : {:f}".format(error_count / float(test_num_of_digits)))

    endtime = datetime.datetime.now()
    print("time of kNN-digits:{}".format((endtime - starttime).seconds))