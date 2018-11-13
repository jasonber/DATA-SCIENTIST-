import tree
import tree_plotter
dataset, labels = tree.create_dataset()
tree.entropy(dataset)

dataset[0][-1] = 'maybe'

tree.choose_best_feature_to_split(dataset)

my_tree = tree.create_tree(dataset, labels)

tree_plotter.retrieve_tree(1)

my_tree = tree_plotter.retrieve_tree(0)

tree_plotter.get_num_leafs(my_tree)

tree_plotter.get_tree_depth(my_tree)