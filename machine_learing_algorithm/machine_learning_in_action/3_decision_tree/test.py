import tree
import tree_plotter
# dataset, labels = tree.create_dataset()
# tree.entropy(dataset)
#
# dataset[0][-1] = 'maybe'
#
# tree.choose_best_feature_to_split(dataset)
#
# my_tree = tree.create_tree(dataset, labels)
#
# tree_plotter.retrieve_tree(1)

my_tree = tree_plotter.retrieve_tree(0)
#
# tree_plotter.get_num_leafs(my_tree)
#
# tree_plotter.get_tree_depth(my_tree)
tree_plotter.create_plot(my_tree)

data, labels = tree.create_dataset()

tree.classify(my_tree, labels, [1, 0])

tree.classify(my_tree, labels, [1, 1])

tree.store_tree('my_tree', "/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                         "machine_learing_algorithm/machine_learning_in_action/3_decision_tree/classifierStorage.txt")

tree.load_tree("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                         "machine_learing_algorithm/machine_learning_in_action/3_decision_tree/classifierStorage.txt")