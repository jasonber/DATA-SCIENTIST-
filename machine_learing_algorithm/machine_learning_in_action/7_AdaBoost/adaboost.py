import numpy as np


def load_simple_data():
    data_matrix = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels

def stump_classify(data_matrix, dimen, threshold, thresh_ineq):
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == "lt":
        ret_array[data_matrix[:, dimen] <= threshold] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > threshold] = -1.0
    return ret_array

def build_stump(data_array, class_label, D):
    data_matrix = np.mat(data_array)
    label_matrix = np.mat(class_label).T
    row, column = np.shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(column):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                threshold = (range_min + float(j) * step_size)
                predict_value = stump_classify(data_matrix, i , threshold, inequal)
                error_array = np.mat(np.ones((m, 1)))
                error_array[predict_value == label_matrix] = 0
                weight_error = D.T * error_array
                print("split: dim {}, threshold {.2f}, "
                      "thresh_inequal: {}, the weight error is {.3f}".format(i, threshold, inequal, weight_error))
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predict_value.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est
