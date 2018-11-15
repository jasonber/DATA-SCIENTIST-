"""
1. ID3 是信息增益分支
2. C4.5 是信息增益率分支
3 .CART 做分类工作时，采用 GINI 值作为节点分裂的依据；
    回归时，采用样本的最小方差作为节点的分裂依据。
所有算法的目的均是使树向着减小数据混乱程度的方向分支
工程上总的来说:
CART 和 C4.5 之间主要差异在于分类结果上，CART 可以回归分析也可以分类，C4.5 只能做分类；
C4.5 子节点是可以多分的，而 CART 是无数个二叉子节点；
以此拓展出以 CART 为基础的 “树群” Random forest ， 以 回归树 为基础的 “树群” GBDT
"""
import numpy as np


def load_dataset(path):
    data_list = []
    fr = open(path)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        # 这里会使create_tree()报错TypeError: list indices must be integers or slices, not tuple。
        # 这是因为dataset是个list不是matrix。
        # 使用append默认时存成List类型，而List类型是无法执行[:,j]这种提取操作，故将其转为Mat类型即可。
        #flt_line = map(float, cur_line)
        flt_line = list(map(float, cur_line))
        data_list.append(flt_line)
    # data_matrix = np.mat(data_list)
    return data_list


def bin_split_dataset(dataset, feature, value):
    mat0 = dataset[np.nonzero(dataset[:,  feature] > value)[0],  :]
    mat1 = dataset[np.nonzero(dataset[:,  feature] <= value)[0],  :]
    return mat0, mat1


def reg_leaf(dataset):
    # 使用叶节点的均值代表叶节点中的值，f(leaf_value) = mean(leaf)
    return np.mean(dataset[:, -1])

def reg_err(dataset):
    # np.var计算出的是方差，需要变换成总体的差异
    return np.var(dataset[:, -1]) * np.shape(dataset)[0]


def choose_best_split(dataset, leaf_type=reg_leaf,  err_type=reg_err,  stop=(1, 4)):
    """
    ops=(1,4)，超参数，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
     之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
     最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
     """
    # 最小误差下降的值小于1 ，停止分支
    stop_s = stop[0]
    #  分支的数据集size小于4，停止分支
    stop_number = stop[1]

    # 如何这个节点内数据size为1，那么就不继续分支了。没有切分特征的返回，只返回切分的最优值，对于叶节点来说就是它们所代表的类别
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(dataset)
    m, n = np.shape(dataset)
    sst = err_type(dataset)
    # 因为要向着减小数据混乱程度的方向分支，所以要初始化数据的混乱程度为无线大
    best_s, best_index, best_value = np.inf, 0, 0
    for feature_index in range(n-1):
        for split_value in set(dataset[:, feature_index].T.tolist()[0]):
            mat0, mat1 = bin_split_dataset(dataset, feature_index, split_value)
            if (np.shape(mat0)[0] < stop_number) or (np.shape(mat1)[0] < stop_number):
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feature_index
                best_value = split_value
                best_s = new_s
    if (sst - best_s) < stop_s:
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_dataset(dataset, best_index, best_value)
    if (np.shape(mat0)[0] < stop_number) or (np.shape(mat1)[0] < stop_number):
        return None, leaf_type(dataset)
    return best_index, best_value


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """

    :param dataset:
    :param leaf_type: 建立叶子节点的函数，方式
    :param err_type: 误差计算方法，函数
    :param ops: 容许误差下降值，切分的最少样本数
    :return:
    """
    feature, value = choose_best_split(dataset, leaf_type, err_type, ops)
    if feature == None:
        return value
    ret_tree = {}
    ret_tree['split_index'] = feature
    ret_tree['split_value'] = value
    left_branch, right_branch = bin_split_dataset(dataset, feature, value)
    ret_tree['left'] = create_tree(left_branch, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(right_branch, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return (type(obj).__name__=='dict')


def get_mean(tree):
    if is_tree(tree['right']): tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']): tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    if (is_tree(tree['right']) or is_tree(tree['left'])):
        left_leaf, right_leaf = bin_split_dataset(test_data, tree['split_index'], tree['split_value'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], left_leaf)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], right_leaf)
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left_leaf, right_leaf = bin_split_dataset(test_data, tree['split_index'], tree['split_value'])
        error_no_merge = np.sum(np.power(left_leaf[:, -1] - tree['left'], 2)) + \
                         np.sum(np.power(right_leaf[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print('merging')
            return tree_mean
        else:
            return tree
    else:
        return tree
