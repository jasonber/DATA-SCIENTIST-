import tree
dataset, labels = tree.create_dataset()
tree.entropy(dataset)

dataset[0][-1] = 'maybe'

tree.choose_best_feature_to_split(dataset)

my_tree = tree.create_tree(dataset, labels)