import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node("decision_node", (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node("leaf_node", (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leafs(my_tree):
    num_leafs = 0
    # my_tree的每一层key是节点，第一个节点就是第一层的key，第一层只有一个key
    first_node = my_tree.keys()
    # Python3中使用LIST转换firstnode，原书使用[0]直接索引只能用于Python2
    first_node = list(first_node)[0]
    # 这里得到了下一个节点的信息
    second_dict = my_tree[first_node]
    for key in second_dict.keys():
        # 不懂__name__。判断first_node的value是否是字典，如果是字典说明tree还在继续分列，需要递归。直到second_dict是一个值。
        # 获取类型的字符串,type（）返回的是type，type（）.__name__返回的是字符串
        if type(second_dict[key]).__name__=="dict":
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_node = my_tree.keys()
    first_node = list(first_node)[0]
    second_dict = my_tree[first_node]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=="dict":
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_trees =  [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1:'yes'}}}},
                      {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]
