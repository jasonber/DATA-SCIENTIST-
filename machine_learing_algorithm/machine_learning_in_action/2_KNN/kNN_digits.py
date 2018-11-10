import numpy as np
import operator
from os import listdir
import kNN
# 为什么出错
# TypeError: hand_write_class_test() missing 1 required positional argument: 'self'
class KnnDigits(object):
    def __int__(self):
        pass

    def img_to_vector(self, filename):
        return_vector = np.zeros(1, 1024)
        file = open(filename)
        for i in range(32):
            line = file.readline()
            for j in range(32):
                return_vector[0, 32 * i + j] = int(line[j])
         return return_vector

    def hand_write_class_test(self):
        hand_write_labels = []
        train_file_list = listdir("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
                                  "machine_learning_in_action/2_KNN/digits/trainingDigits")
        train_num_of_digits = len(train_file_list)
        train_data = np.zeros((train_num_of_digits, 1024))
        for i in range(train_num_of_digits):
            file_name_str = train_file_list[i]
            file_str = file_name_str.split('.')[0]
            class_digits_str = int(file_str.split('_')[0])
            hand_write_labels.append(class_digits_str)
            train_data[i, :] = self.img_to_vector('train_Digits {}'.format(file_name_str))

        test_file_list = listdir("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
                                 "machine_learning_in_action/2_KNN/digits/testDigits")
        error_count = 0.0
        test_num_of_digits = len(test_file_list)
        for i in range(test_num_of_digits):
            file_name_str = test_file_list[i]
            file_str = file_name_str.split('.')[0]
            class_digits_str = int(file_str.split('_')[0])
            test_data = self.img_to_vector('test_Digits {}'.format(file_name_str))
            result = kNN.classify0(test_data, train_data, hand_write_labels, 3)
            print("the classifier came back with: {:d}, the real answer is: {:d}".format(result, class_digits_str))
            if (result != class_digits_str):
                error_count += 1.0
        print("the total number of errors is {:d}".format(error_count))
        print("the total number of error rate is : {:f}".format(error_count / float(test_num_of_digits)))
