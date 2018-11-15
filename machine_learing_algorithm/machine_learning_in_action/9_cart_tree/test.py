import reg_tree
import numpy as np
import matplotlib.pyplot as plt

test_matrix = np.mat(np.eye(4))

mat0, mat1 = reg_tree.bin_split_dataset(test_matrix, 1, 0.5)

dataset = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
                                "machine_learning_in_action/9_cart_tree/ex00.txt")
dataset_matrix = np.mat(dataset)
reg_tree.create_tree(dataset_matrix)
dataset_x = []
dataset_y = []
for x, y in dataset:
    dataset_x.append(x)
    dataset_y.append(y)

fig = plt.figure(figsize=(6, 8))
plt.scatter(dataset_x, dataset_y)
plt.show()



dataset2 = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm"
                                 "/machine_learning_in_action/9_cart_tree/ex0.txt")
dataset2_matrix = np.mat(dataset2)
my_tree = reg_tree.create_tree(dataset2_matrix, ops=(0,1))

# dataset2_x = []
# dataset2_y = []
# for a,x, y in dataset2:
#     dataset2_x.append(x)
#     dataset2_y.append(y)
# plt.scatter(dataset2_x, dataset_y)

dataset3 = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
                                 "machine_learning_in_action/9_cart_tree/ex2.txt")

dataset3_matrix = np.mat(dataset3)
tree = reg_tree.prune(my_tree, dataset3_matrix)


