import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制带箭头的注释
    :param node_txt: 箭头名称
    :param center_pt: 开始位置
    :param parent_pt: 终止位置
    :param node_type: 节点的类型
    :return:
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


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
        if type(second_dict[key]).__name__ == "dict":
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
        if type(second_dict[key]).__name__ == "dict":
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parent_pt, text_string):
    """
    计算tree的中间位置
    :param cntr_pt: 起始位置，子节点坐标
    :param parent_pt: 终止位置，父节点坐标
    :param text_string: 文本标签信息
    :return:
    """
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, text_string)


def plot_tree(my_tree, parent_pt, node_text):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_node = my_tree.keys()
    first_node = list(first_node)[0]
    cntr_pt = (plot_tree.xoff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalw, plot_tree.yoff)
    plot_mid_text(cntr_pt, parent_pt, node_text)
    plot_node(first_node, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_node]
    plot_tree.yoff = plot_tree.yoff - 1.0 / plot_tree.totald
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xoff = plot_tree.xoff + 1.0 / plot_tree.totalw
            plot_node(second_dict[key], (plot_tree.xoff, plot_tree.yoff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntr_pt, str(key))
    plot_tree.yoff = plot_tree.yoff + 1.0 / plot_tree.totald


def create_plot(input_tree):
    # 定义一个画布，背景为白色
    fig = plt.figure(1, facecolor='white')
    # 画布清空
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    # 之前的代码
    # create_plot.ax1 = plt.subplot(111, frameon=False)
    # plot_node("decision_node", (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node("leaf_node", (0.8, 0.1), (0.3, 0.8), leaf_node)
    # plt.show()

    # create_polt.ax1 是全局变量，绘制图像的句柄， subplot定义了一个绘图
    # 111表示把画布分成了1行1列，即只有1个图， 最后1个1表示这个画布上的第一个图
    # frameon 表示是否绘制坐标轴矩形
    create_plot.ax1 = plt.subplot(111, frameon=False, **ax_props)

    plot_tree.totalw = float(get_num_leafs(input_tree))
    plot_tree.totald = float(get_tree_depth(input_tree))
    plot_tree.xoff = -0.6 / plot_tree.totalw
    plot_tree.yoff = 1.2
    plot_tree(input_tree, (0.5, 1.0), '')
    plt.show()
